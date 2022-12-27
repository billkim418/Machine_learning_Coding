# Chapter 5 : Semi-supervised Learning

이번 튜토리얼 에서는 앙상블(Ensemble)이라는 주제를 다뤄 보겠습니다. 앙상블이란 지금까지 튜토리얼에서 다뤄 본 여러 머신 러닝 알고리즘들을 다양한 방식으로 결합 혹은 여러번 학습함으로써 성능을 올리는 방법론입니다. 특히 우리는 그 중에서 가장 유명한 모델인 XGBoost를 소개하고 이를 생존분석과 연결 이를 실제 데이터셋은 METBRIC(유방암 데이터셋)과의 적용을 해보겠습니다.
추가적으로 해당 레포는 [고려대학교 강필성 교수님](https://github.com/pilsung-kang)의 수업을 듣고 작성했음을 밝힙니다.

--- 
## 준지도 학습(Semi-supervised learning)이란?
현실에서 데이터에 대한 답이 있는 경우를 예측할 때를 supervised learning이라 하고, 답이 없는 경우는 unsupervied learning이라고 합니다. 하지만 현실 데이터의 경우 모든 데이터에 대한 라벨 즉 정답이 없는 경우가 많습니다.이러한 경우 라벨이 없는 데이터를 버리기에는 아깝기 때문에 해당 데이터를 같이 사용하여 학습햐는 방법론을 Semi-supervised learning이라고 합니다. <br>

## Deep Semi-supervised Learning
- 준지도 학습의 종류에는 아래와 같은 모델들이 있는데 해당 튜토리얼 에서는 Consistency regularization 위주로 구현 및 비교해보겠습니다.
![image](https://user-images.githubusercontent.com/68594529/209619297-a61c375f-97e6-4be9-ad69-18325a80e99f.png)

## Consistency regularizatio
- Unlabeled data에 작은 변화를 주어도 예측의 결과에는 일관성이 있을거라는 가정에서 출발하였습니다.
- Unlabeded data는 예측결과를 알 수 없기 때문에 data augmentation을 통해 class가 바뀌지 않을 정도의 변화를 줬을 때, 원래 데이터와의 예측결과가 같아지도록 unsupervised loss를 주어 학습시키는게 주된 방법입니다.

## Ladder network
- 지도학습과 비지도학습을 결합한 대표적인 모형입니다.
- Hierarchical latent variable model의 계층적인 특징을 반영한 오토 인코더를 사용합니다.
- 또한, 효과적인 학습을 위하여 Denosing 기법을 활용하는데 가우시안 노이지를 주로 추가하여 학습 데이터의 특징을 더 잘 반영합니다.

### Ladder network의 구조
![image](https://user-images.githubusercontent.com/68594529/209623580-2bf87422-a8af-4f25-b2a0-be3e4483ad4d.png)<br>
- 해당 그림의 왼쪽을 보면 가우시안 노이즈가 추가 됨을 알 수 있습니다.
- 또한 왼쪽의 인코더를 corrupted encoder라고 부르며 인코더와 디코더간의 결과가 일치하게 학습됨을 알수 있고, 해당 결과 즉 latent vector들이 오른쪽의 clean encoder를 통해 나온 값과 일치하게끔 학습되고 있습니다.

### Ladder netwrok 실행 코드

```python
class AddBeta(Layer):
    def __init__(self  , **kwargs):
        super(AddBeta, self).__init__(**kwargs)
        
    def build(self, input_shape):
        if self.built:
            return
        
        self.beta = self.add_weight(name='beta', 
                                      shape= input_shape[1:] ,
                                      initializer='zeros',
                                      trainable=True)
       
        self.built = True
        super(AddBeta, self).build(input_shape)  
        
    def call(self, x, training=None):
        return tf.add(x, self.beta)


class G_Guass(Layer):
    def __init__(self , **kwargs):
        super(G_Guass, self).__init__(**kwargs)

    def wi(self, init, name):
        if init == 1:
            return self.add_weight(name='guess_'+name, 
                                      shape=(self.size,),
                                      initializer='ones',
                                      trainable=True)
        elif init == 0:
            return self.add_weight(name='guess_'+name, 
                                      shape=(self.size,),
                                      initializer='zeros',
                                      trainable=True)
        else:
            raise ValueError("Invalid argument '%d' provided for init in G_Gauss layer" % init)


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.size = input_shape[0][-1]

        init_values = [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.]
        self.a = [self.wi(v, 'a' + str(i + 1)) for i, v in enumerate(init_values)]
        super(G_Guass , self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        z_c, u = x 

        def compute(y):
            return y[0] * tf.sigmoid(y[1] * u + y[2]) + y[3] * u + y[4]

        mu = compute(self.a[:5])
        v  = compute(self.a[5:])

        z_est = (z_c - mu) * v + mu
        return z_est
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.size)


def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))


def add_noise( inputs , noise_std ):
    return Lambda( lambda x: x + tf.random.normal(tf.shape(x)) * noise_std  )( inputs )


def get_ladder_network_fc(layer_sizes=[784, 1000, 500, 250, 250, 250, 10], 
     noise_std=0.3,
     denoising_cost=[1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]):

    L = len(layer_sizes) - 1  # number of layers

    inputs_l = Input((layer_sizes[0],))  
    inputs_u = Input((layer_sizes[0],))  

    fc_enc = [Dense(s, use_bias=False, kernel_initializer='glorot_normal') for s in layer_sizes[1:] ]
    fc_dec = [Dense(s, use_bias=False, kernel_initializer='glorot_normal') for s in layer_sizes[:-1]]
    betas  = [AddBeta() for l in range(L)]

    def encoder(inputs, noise_std  ):
        h = add_noise(inputs, noise_std)
        all_z    = [None for _ in range( len(layer_sizes))]
        all_z[0] = h
        
        for l in range(1, L+1):
            z_pre = fc_enc[l-1](h)
            z =     Lambda(batch_normalization)(z_pre) 
            z =     add_noise (z, noise_std)
            
            if l == L:
                h = Activation('softmax')(betas[l-1](z))
            else:
                h = Activation('relu')(betas[l-1](z))
                
            all_z[l] = z

        return h, all_z

    y_c_l, _ = encoder(inputs_l, noise_std)
    y_l, _   = encoder(inputs_l, 0.0)  

    y_c_u, corr_z  = encoder(inputs_u , noise_std)
    y_u,  clean_z  = encoder(inputs_u , 0.0)  

    # Decoder
    d_cost = []  # to store the denoising cost of all layers
    for l in range(L, -1, -1):
        z, z_c = clean_z[l], corr_z[l]
        if l == L:
            u = y_c_u
        else:
            u = fc_dec[l]( z_est ) 
        u = Lambda(batch_normalization)(u)
        z_est  = G_Guass()([z_c, u])  
        d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est - z), 1)) / layer_sizes[l]) * denoising_cost[l])

    u_cost = tf.add_n(d_cost)

    y_c_l = Lambda(lambda x: x[0])([y_c_l, y_l, y_c_u, y_u, u, z_est, z])

    tr_m = Model([inputs_l, inputs_u], y_c_l)
    tr_m.add_loss(u_cost)
    tr_m.compile(keras.optimizers.Adam(lr=0.02 ), 'categorical_crossentropy', metrics=['accuracy'])


    tr_m.metrics_names.append("den_loss")
    tr_m.metrics_tensors.append(u_cost)

    te_m = Model(inputs_l, y_l)
    tr_m.test_model = te_m

    return tr_m
```
## PI-MODEL
![image](https://user-images.githubusercontent.com/68594529/209650685-29701e0e-54ce-4b46-9245-3c5aa7af485c.png)
- 하나의 모델에 dropout을 적용한 2가지 방식으로 학습을 진행합니다. 이 과정에서 적용되는 augmentation은 동일합니다.
- 같은 모델을 사용하지만 각각 다른 방식으로 dropout이 적용되어 supervised loss 같은 경우 cross-entropy로 계산합니다.
- 또한 unsupervised loss 같은 경우에는 모델을 통해 나온 결과물의 squared difference를 계산합니다.
- 따라서 supervised, unsupervised 두가지 모두 최소화하는 모델입니다.

### PI-MODEL 실행 코드

```python
class PI(object):

    def __init__(self, d, lr, lambda_pi_usl, use_pi):

        """ flags for each regularizor """
        self.use_pi   = use_pi

        """ data and external toolkits """
        self.d  = d  # dataset manager
        self.ls = Layers()
        self.lf = LossFunctions(self.ls, d, self.encoder)

        """ placeholders defined outside"""
        self.lr  = lr
        self.lambda_pi_usl = lambda_pi_usl	

    def encoder(self, x, is_train=True, do_update_bn=True):

        """ https://arxiv.org/pdf/1610.02242.pdf """

        if is_train:
            h = self.distort(x)
            h = self.ls.get_corrupted(x, 0.15)
        else:
            h = x

        scope = '1'
        h = self.ls.conv2d(scope+'_1', h, 128, activation=self.ls.lrelu)
        h = self.ls.conv2d(scope+'_2', h, 128, activation=self.ls.lrelu)
        h = self.ls.conv2d(scope+'_3', h, 128, activation=self.ls.lrelu)
        h = self.ls.max_pool(h)
        if is_train: h = tf.nn.dropout(h, 0.5)

        scope = '2'
        h = self.ls.conv2d(scope+'_1', h, 256, activation=self.ls.lrelu)
        h = self.ls.conv2d(scope+'_2', h, 256, activation=self.ls.lrelu)
        h = self.ls.conv2d(scope+'_3', h, 256, activation=self.ls.lrelu)
        h = self.ls.max_pool(h)
        if is_train: h = tf.nn.dropout(h, 0.5)

        scope = '3'
        h = self.ls.conv2d(scope+'_1', h, 512, activation=self.ls.lrelu)
        h = self.ls.conv2d(scope+'_2', h, 256, activation=self.ls.lrelu, filter_size=(1,1))
        h = self.ls.conv2d(scope+'_3', h, 128, activation=self.ls.lrelu, filter_size=(1,1))
        h = tf.reduce_mean(h, reduction_indices=[1, 2])  # Global average pooling
        h = self.ls.dense(scope, h, self.d.l)

        return h

    def build_graph_train(self, x_l, y_l, x, is_supervised=True):

        o = dict()  # output
        loss = 0

        logit = self.encoder(x)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            logit_l = self.encoder(x_l, is_train=True, do_update_bn=False)  # for pyx and vat loss computation

        """ Classification Loss """
        o['Ly'], o['accur'] = self.lf.get_loss_pyx(logit_l, y_l)
        loss += o['Ly']

        """ PI Model Loss """
        if self.use_pi:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                _,_,o['Lp'] = self.lf.get_loss_pi(x, logit, is_train=True)
                loss += self.lambda_pi_usl * o['Lp']
        else:
            o['Lp'] = tf.constant(0)

        """ set losses """
        o['loss'] = loss
        self.o_train = o

        """ set optimizer """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
        #self.op = optimizer.minimize(loss)
        grads = optimizer.compute_gradients(loss)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                #g = tf.Print(g, [g], "g %s = "%(v))
                grads[i] = (tf.clip_by_norm(g,5),v) # clip gradients
            else:
                print('g is None:', v)
                v = tf.Print(v, [v], "v = ", summarize=10000)
        self.op = optimizer.apply_gradients(grads) # return train_op


    def build_graph_test(self, x_l, y_l ):

        o = dict()  # output
        loss = 0

        logit_l = self.encoder(x_l, is_train=False, do_update_bn=False)  # for pyx and vat loss computation

        """ classification loss """
        o['Ly'], o['accur'] = self.lf.get_loss_pyx(logit_l, y_l)
        loss += o['Ly']

        """ set losses """
        o['loss'] = loss
        self.o_test = o

    def distort(self, x):
    
        _d = self.d

        def _distort(a_image):
            """
            bounding_boxes: A Tensor of type float32.
                3-D with shape [batch, N, 4] describing the N bounding boxes associated with the image. 
            Bounding boxes are supplied and returned as [y_min, x_min, y_max, x_max]
            """
            # shape: [1, 1, 4]
            bounding_boxes = tf.constant([[[1/10, 1/10, 9/10, 9/10]]], dtype=tf.float32)
                                                                                                         
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                                (_d.h, _d.w, _d.c), bounding_boxes,
                                min_object_covered=(8.5/10.0),
                                aspect_ratio_range=[7.0/10.0, 10.0/7.0])

            a_image = tf.slice(a_image, begin, size)
            """ for the purpose of distorting not use tf.image.resize_image_with_crop_or_pad under """
            a_image = tf.image.resize_images(a_image, [_d.h, _d.w])
            """ due to the size of channel returned from tf.image.resize_images is not being given,
                specify it manually. """
            a_image = tf.reshape(a_image, [_d.h, _d.w, _d.c])
            return a_image

        """ process batch times in parallel """
        return tf.map_fn( _distort, x)
```
## Temporal Ensemble
![image](https://user-images.githubusercontent.com/68594529/209661674-98fc642b-3009-45e2-869c-9c06883de87b.png)<br>
- PI model을 통해 얻어진 타겟 값은 노이즈가 있기 때문에 temoporal ensemble은 이전 network들의 결과들을 사용함으로써 이러한 노이즈들의 문제를 완화시킵니다.
- 또한 PI model과는 달리 이전 ensemble의 결과들을 unsupervised의 target으로 사용하고 첫 Epoch에는 ensemble output이 없기 때문에 가중치(w)와 함께 보통 0으로 설정합니다.
- 앙상블의 효과를 통해 기존 pi model의 노이즈한 부분들을 제거하고 성능을 올려줄수 있습니다.


### Temporal Ensemble 실행 코드

```python
def sample_train(train_dataset, test_dataset, batch_size, k, n_classes,
                 seed, shuffle_train=True, return_idxs=True):
    
    n = len(train_dataset)
    rrng = np.random.RandomState(seed)
    
    cpt = 0
    indices = torch.zeros(k)
    other = torch.zeros(n - k)
    card = k // n_classes
    
    for i in xrange(n_classes):
        class_items = (train_dataset.train_labels == i).nonzero()
        n_class = len(class_items)
        rd = np.random.permutation(np.arange(n_class))
        indices[i * card: (i + 1) * card] = class_items[rd[:card]]
        other[cpt: cpt + n_class - card] = class_items[rd[card:]]
        cpt += n_class - card

    other = other.long()
    train_dataset.train_labels[other] = -1

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size,
                                              num_workers=4,
                                              shuffle=False)
    
    if return_idxs:
        return train_loader, test_loader, indices 
    return train_loader, test_loader


def temporal_loss(out1, out2, w, labels):
    
    # MSE between current and temporal outputs
    def mse_loss(out1, out2):
        quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
        return quad_diff / out1.data.nelement()
    
    def masked_crossentropy(out, labels):
        cond = (labels >= 0)
        nnz = torch.nonzero(cond)
        nbsup = len(nnz)
        # check if labeled samples in batch, return 0 if none
        if nbsup > 0:
            masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
            masked_labels = labels[cond]
            loss = F.cross_entropy(masked_outputs, masked_labels)
            return loss, nbsup
        return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=False), 0
    
    sup_loss, nbsup = masked_crossentropy(out1, labels)
    unsup_loss = mse_loss(out1, out2)
    return sup_loss + w * unsup_loss, sup_loss, unsup_loss, nbsup


def train(model, seed, k=100, alpha=0.6, lr=0.002, beta2=0.99, num_epochs=150,
          batch_size=100, drop=0.5, std=0.15, fm1=16, fm2=32,
          divide_by_bs=False, w_norm=False, data_norm='pixelwise',
          early_stop=None, c=300, n_classes=10, max_epochs=80,
          max_val=30., ramp_up_mult=-5., n_samples=60000,
          print_res=True, **kwargs):
    
    # retrieve data
    train_dataset, test_dataset = prepare_mnist()
    ntrain = len(train_dataset)

    # build model
    model.cuda()

    # make data loaders
    train_loader, test_loader, indices = sample_train(train_dataset, test_dataset, batch_size,
                                                      k, n_classes, seed, shuffle_train=False)

    # setup param optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # train
    model.train()
    losses = []
    sup_losses = []
    unsup_losses = []
    best_loss = 20.

    Z = torch.zeros(ntrain, n_classes).float().cuda()        # intermediate values
    z = torch.zeros(ntrain, n_classes).float().cuda()        # temporal outputs
    outputs = torch.zeros(ntrain, n_classes).float().cuda()  # current outputs

    for epoch in range(num_epochs):
        t = timer()
        
        # evaluate unsupervised cost weight
        w = weight_schedule(epoch, max_epochs, max_val, ramp_up_mult, k, n_samples)
     
        if (epoch + 1) % 10 == 0:
            print 'unsupervised loss weight : {}'.format(w)
        
        # turn it into a usable pytorch object
        w = torch.autograd.Variable(torch.FloatTensor([w]).cuda(), requires_grad=False)
        
        l = []
        supl = []
        unsupl = []
        for i, (images, labels) in enumerate(train_loader):  
            images = Variable(images.cuda())
            labels = Variable(labels.cuda(), requires_grad=False)

            # get output and calculate loss
            optimizer.zero_grad()
            out = model(images)
            zcomp = Variable(z[i * batch_size: (i + 1) * batch_size], requires_grad=False)
            loss, suploss, unsuploss, nbsup = temporal_loss(out, zcomp, w, labels)

            # save outputs and losses
            outputs[i * batch_size: (i + 1) * batch_size] = out.data.clone()
            l.append(loss.data[0])
            supl.append(nbsup * suploss.data[0])
            unsupl.append(unsuploss.data[0])

            # backprop
            loss.backward()
            optimizer.step()

            # print loss
            if (epoch + 1) % 10 == 0:
                if i + 1 == 2 * c:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.6f, Time (this epoch): %.2f s' 
                           %(epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, np.mean(l), timer() - t))
                elif (i + 1) % c == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.6f' 
                           %(epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, np.mean(l)))

        # update temporal ensemble
        Z = alpha * Z + (1. - alpha) * outputs
        z = Z * (1. / (1. - alpha ** (epoch + 1)))

        # handle metrics, losses, etc.
        eloss = np.mean(l)
        losses.append(eloss)
        sup_losses.append((1. / k) * np.sum(supl))  # division by 1/k to obtain the mean supervised loss
        unsup_losses.append(np.mean(unsupl))
        
        # saving model 
        if eloss < best_loss:
            best_loss = eloss
            torch.save({'state_dict': model.state_dict()}, 'model_best.pth.tar')

    # test
    model.eval()
    acc = calc_metrics(model, test_loader)
    if print_res:
        print 'Accuracy of the network on the 10000 test images: %.2f %%' % (acc)
        
    # test best model
    checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    acc_best = calc_metrics(model, test_loader)
    if print_res:
        print 'Accuracy of the network (best model) on the 10000 test images: %.2f %%' % (acc_best)
     
    return acc, acc_best, losses, sup_losses, unsup_losses, indices
```

## Mean Teacher
![image](https://user-images.githubusercontent.com/68594529/209662400-9e28f0e3-8493-4c1c-b5f2-d1c74b7d6812.png)<br>
- 기존의 파이 모델은 하나의 모델이 teacher, student 두 가지의 역활을 수행하는 단점이 존재합니다.
- 또한 Epoch당 개선이 이루어지므로 해당 부분에서 학습속도가 느리다는 단점을 가지고 있습니다.
- 이러한 부분을 개선한 모델이 Mean teacher 모델입니다. 

### Mean Teacher의 구조
![image](https://user-images.githubusercontent.com/68594529/209663187-417da39f-1514-4a83-99da-7fb10bfaef1e.png)<br>
- 해당 모델의 구조를 보면 latent 한 vector들이 업데이트 대상임을 확인할 수 있다. 이는 기존의 PI 모델의 업데이트 대상인 Output 과는 다른것을 확인할 수 있습니다.
![image](https://user-images.githubusercontent.com/68594529/209666664-78c9f10a-6e53-460c-afdf-ce934bb81737.png)
- loss를 살펴보면 마찬가지로 Unsepervised와 supervised loss의 합으로 define되지만 아직 consistency의 개념이 loss에 적용되지 않았음을 확인할수 있습니다.

### Mean Teacher 실행 코드
```python
class Model:
    DEFAULT_HYPERPARAMS = {
        # Consistency hyperparameters
        'ema_consistency': True,
        'apply_consistency_to_labeled': True,
        'max_consistency_cost': 100.0,
        'ema_decay_during_rampup': 0.99,
        'ema_decay_after_rampup': 0.999,
        'consistency_trust': 0.0,
        'num_logits': 1, # Either 1 or 2
        'logit_distance_cost': 0.0, # Matters only with 2 outputs

        # Optimizer hyperparameters
        'max_learning_rate': 0.003,
        'adam_beta_1_before_rampdown': 0.9,
        'adam_beta_1_after_rampdown': 0.5,
        'adam_beta_2_during_rampup': 0.99,
        'adam_beta_2_after_rampup': 0.999,
        'adam_epsilon': 1e-8,

        # Architecture hyperparameters
        'input_noise': 0.15,
        'student_dropout_probability': 0.5,
        'teacher_dropout_probability': 0.5,

        # Training schedule
        'rampup_length': 40000,
        'rampdown_length': 25000,
        'training_length': 150000,

        # Input augmentation
        'flip_horizontally': False,
        'translate': True,

        # Whether to scale each input image to mean=0 and std=1 per channel
        # Use False if input is already normalized in some other way
        'normalize_input': True,

        # Output schedule
        'print_span': 20,
        'evaluation_span': 500,
    }

  
    def __init__(self, run_context=None):
        if run_context is not None:
            self.training_log = run_context.create_train_log('training')
            self.validation_log = run_context.create_train_log('validation')
            self.checkpoint_path = os.path.join(run_context.transient_dir, 'checkpoint')
            self.tensorboard_path = os.path.join(run_context.result_dir, 'tensorboard')

        with tf.name_scope("placeholders"):
            self.images = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='images')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        tf.add_to_collection("init_in_init", self.global_step)
        self.hyper = HyperparamVariables(self.DEFAULT_HYPERPARAMS)
        for var in self.hyper.variables.values():
            tf.add_to_collection("init_in_init", var)

        with tf.name_scope("ramps"):
            sigmoid_rampup_value = sigmoid_rampup(self.global_step, self.hyper['rampup_length'])
            sigmoid_rampdown_value = sigmoid_rampdown(self.global_step,
                                                      self.hyper['rampdown_length'],
                                                      self.hyper['training_length'])
            self.learning_rate = tf.multiply(sigmoid_rampup_value * sigmoid_rampdown_value,
                                             self.hyper['max_learning_rate'],
                                             name='learning_rate')
            self.adam_beta_1 = tf.add(sigmoid_rampdown_value * self.hyper['adam_beta_1_before_rampdown'],
                                      (1 - sigmoid_rampdown_value) * self.hyper['adam_beta_1_after_rampdown'],
                                      name='adam_beta_1')
            self.cons_coefficient = tf.multiply(sigmoid_rampup_value,
                                                self.hyper['max_consistency_cost'],
                                                name='consistency_coefficient')

            step_rampup_value = step_rampup(self.global_step, self.hyper['rampup_length'])
            self.adam_beta_2 = tf.add((1 - step_rampup_value) * self.hyper['adam_beta_2_during_rampup'],
                                      step_rampup_value * self.hyper['adam_beta_2_after_rampup'],
                                      name='adam_beta_2')
            self.ema_decay = tf.add((1 - step_rampup_value) * self.hyper['ema_decay_during_rampup'],
                                    step_rampup_value * self.hyper['ema_decay_after_rampup'],
                                    name='ema_decay')

        (
            (self.class_logits_1, self.cons_logits_1),
            (self.class_logits_2, self.cons_logits_2),
            (self.class_logits_ema, self.cons_logits_ema)
        ) = inference(
            self.images,
            is_training=self.is_training,
            ema_decay=self.ema_decay,
            input_noise=self.hyper['input_noise'],
            student_dropout_probability=self.hyper['student_dropout_probability'],
            teacher_dropout_probability=self.hyper['teacher_dropout_probability'],
            normalize_input=self.hyper['normalize_input'],
            flip_horizontally=self.hyper['flip_horizontally'],
            translate=self.hyper['translate'],
            num_logits=self.hyper['num_logits'])

        with tf.name_scope("objectives"):
            self.mean_error_1, self.errors_1 = errors(self.class_logits_1, self.labels)
            self.mean_error_ema, self.errors_ema = errors(self.class_logits_ema, self.labels)

            self.mean_class_cost_1, self.class_costs_1 = classification_costs(
                self.class_logits_1, self.labels)
            self.mean_class_cost_ema, self.class_costs_ema = classification_costs(
                self.class_logits_ema, self.labels)

            labeled_consistency = self.hyper['apply_consistency_to_labeled']
            consistency_mask = tf.logical_or(tf.equal(self.labels, -1), labeled_consistency)
            self.mean_cons_cost_pi, self.cons_costs_pi = consistency_costs(
                self.cons_logits_1, self.class_logits_2, self.cons_coefficient, consistency_mask, self.hyper['consistency_trust'])
            self.mean_cons_cost_mt, self.cons_costs_mt = consistency_costs(
                self.cons_logits_1, self.class_logits_ema, self.cons_coefficient, consistency_mask, self.hyper['consistency_trust'])


            def l2_norms(matrix):
                l2s = tf.reduce_sum(matrix ** 2, axis=1)
                mean_l2 = tf.reduce_mean(l2s)
                return mean_l2, l2s

            self.mean_res_l2_1, self.res_l2s_1 = l2_norms(self.class_logits_1 - self.cons_logits_1)
            self.mean_res_l2_ema, self.res_l2s_ema = l2_norms(self.class_logits_ema - self.cons_logits_ema)
            self.res_costs_1 = self.hyper['logit_distance_cost'] * self.res_l2s_1
            self.mean_res_cost_1 = tf.reduce_mean(self.res_costs_1)
            self.res_costs_ema = self.hyper['logit_distance_cost'] * self.res_l2s_ema
            self.mean_res_cost_ema = tf.reduce_mean(self.res_costs_ema)

            self.mean_total_cost_pi, self.total_costs_pi = total_costs(
                self.class_costs_1, self.cons_costs_pi, self.res_costs_1)
            self.mean_total_cost_mt, self.total_costs_mt = total_costs(
                self.class_costs_1, self.cons_costs_mt, self.res_costs_1)
            assert_shape(self.total_costs_pi, [3])
            assert_shape(self.total_costs_mt, [3])

            self.cost_to_be_minimized = tf.cond(self.hyper['ema_consistency'],
                                                lambda: self.mean_total_cost_mt,
                                                lambda: self.mean_total_cost_pi)

        with tf.name_scope("train_step"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step_op = nn.adam_optimizer(self.cost_to_be_minimized,
                                                       self.global_step,
                                                       learning_rate=self.learning_rate,
                                                       beta1=self.adam_beta_1,
                                                       beta2=self.adam_beta_2,
                                                       epsilon=self.hyper['adam_epsilon'])

        self.training_control = training_control(self.global_step,
                                                 self.hyper['print_span'],
                                                 self.hyper['evaluation_span'],
                                                 self.hyper['training_length'])

        self.training_metrics = {
            "learning_rate": self.learning_rate,
            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,
            "ema_decay": self.ema_decay,
            "cons_coefficient": self.cons_coefficient,
            "train/error/1": self.mean_error_1,
            "train/error/ema": self.mean_error_ema,
            "train/class_cost/1": self.mean_class_cost_1,
            "train/class_cost/ema": self.mean_class_cost_ema,
            "train/cons_cost/pi": self.mean_cons_cost_pi,
            "train/cons_cost/mt": self.mean_cons_cost_mt,
            "train/res_cost/1": self.mean_res_cost_1,
            "train/res_cost/ema": self.mean_res_cost_ema,
            "train/total_cost/pi": self.mean_total_cost_pi,
            "train/total_cost/mt": self.mean_total_cost_mt,
        }

        with tf.variable_scope("validation_metrics") as metrics_scope:
            self.metric_values, self.metric_update_ops = metrics.aggregate_metric_map({
                "eval/error/1": streaming_mean(self.errors_1),
                "eval/error/ema": streaming_mean(self.errors_ema),
                "eval/class_cost/1": streaming_mean(self.class_costs_1),
                "eval/class_cost/ema": streaming_mean(self.class_costs_ema),
                "eval/res_cost/1": streaming_mean(self.res_costs_1),
                "eval/res_cost/ema": streaming_mean(self.res_costs_ema),
            })
            metric_variables = slim.get_local_variables(scope=metrics_scope.name)
            self.metric_init_op = tf.variables_initializer(metric_variables)

        self.result_formatter = string_utils.DictFormatter(
            order=["eval/error/ema", "error/1", "class_cost/1", "cons_cost/mt"],
            default_format='{name}: {value:>10.6f}',
            separator=",  ")
        self.result_formatter.add_format('error', '{name}: {value:>6.1%}')

        with tf.name_scope("initializers"):
            init_init_variables = tf.get_collection("init_in_init")
            train_init_variables = [
                var for var in tf.global_variables() if var not in init_init_variables
            ]
            self.init_init_op = tf.variables_initializer(init_init_variables)
            self.train_init_op = tf.variables_initializer(train_init_variables)

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.run(self.init_init_op)

    def __setitem__(self, key, value):
        self.hyper.assign(self.session, key, value)

    def __getitem__(self, key):
        return self.hyper.get(self.session, key)

    def train(self, training_batches, evaluation_batches_fn):
        self.run(self.train_init_op, self.feed_dict(next(training_batches)))
        LOG.info("Model variables initialized")
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()
        for batch in training_batches:
            results, _ = self.run([self.training_metrics, self.train_step_op],
                                  self.feed_dict(batch))
            step_control = self.get_training_control()
            self.training_log.record(step_control['step'], {**results, **step_control})
            if step_control['time_to_print']:
                LOG.info("step %5d:   %s", step_control['step'], self.result_formatter.format_dict(results))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                self.evaluate(evaluation_batches_fn)
                self.save_checkpoint()
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()

    def evaluate(self, evaluation_batches_fn):
        self.run(self.metric_init_op)
        for batch in evaluation_batches_fn():
            self.run(self.metric_update_ops,
                     feed_dict=self.feed_dict(batch, is_training=False))
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
        self.validation_log.record(step, results)
        LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))

    def get_training_control(self):
        return self.session.run(self.training_control)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def feed_dict(self, batch, is_training=True):
        return {
            self.images: batch['x'],
            self.labels: batch['y'],
            self.is_training: is_training
        }

    def save_checkpoint(self):
        path = self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def save_tensorboard_graph(self):
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(self.session.graph)
        return writer.get_logdir()


Hyperparam = namedtuple("Hyperparam", ['tensor', 'getter', 'setter'])


def training_control(global_step, print_span, evaluation_span, max_step, name=None):
    with tf.name_scope(name, "training_control"):
        return {
            "step": global_step,
            "time_to_print": tf.equal(tf.mod(global_step, print_span), 0),
            "time_to_evaluate": tf.equal(tf.mod(global_step, evaluation_span), 0),
            "time_to_stop": tf.greater_equal(global_step, max_step),
        }


def step_rampup(global_step, rampup_length):
    result = tf.cond(global_step < rampup_length,
                     lambda: tf.constant(0.0),
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
    global_step = tf.to_float(global_step)
    rampup_length = tf.to_float(rampup_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    global_step = tf.to_float(global_step)
    rampdown_length = tf.to_float(rampdown_length)
    training_length = tf.to_float(training_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
        return tf.exp(-12.5 * phase * phase)

    result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp,
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampdown")


def inference(inputs, is_training, ema_decay, input_noise, student_dropout_probability, teacher_dropout_probability,
              normalize_input, flip_horizontally, translate, num_logits):
    tower_args = dict(inputs=inputs,
                      is_training=is_training,
                      input_noise=input_noise,
                      normalize_input=normalize_input,
                      flip_horizontally=flip_horizontally,
                      translate=translate,
                      num_logits=num_logits)

    with tf.variable_scope("initialization") as var_scope:
        _ = tower(**tower_args, dropout_probability=student_dropout_probability, is_initialization=True)
    with name_variable_scope("primary", var_scope, reuse=True) as (name_scope, _):
        class_logits_1, cons_logits_1 = tower(**tower_args, dropout_probability=student_dropout_probability, name=name_scope)
    with name_variable_scope("secondary", var_scope, reuse=True) as (name_scope, _):
        class_logits_2, cons_logits_2 = tower(**tower_args, dropout_probability=teacher_dropout_probability, name=name_scope)
    with ema_variable_scope("ema", var_scope, decay=ema_decay):
        class_logits_ema, cons_logits_ema = tower(**tower_args, dropout_probability=teacher_dropout_probability, name=name_scope)
        class_logits_ema, cons_logits_ema = tf.stop_gradient(class_logits_ema), tf.stop_gradient(cons_logits_ema)
    return (class_logits_1, cons_logits_1), (class_logits_2, cons_logits_2), (class_logits_ema, cons_logits_ema)


def tower(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          flip_horizontally,
          translate,
          num_logits,
          is_initialization=False,
          name=None):
    with tf.name_scope(name, "tower"):
        default_conv_args = dict(
            padding='SAME',
            kernel_size=[3, 3],
            activation_fn=nn.lrelu,
            init=is_initialization
        )
        training_mode_funcs = [
            nn.random_translate, nn.flip_randomly, nn.gaussian_noise, slim.dropout,
            wn.fully_connected, wn.conv2d
        ]
        training_args = dict(
            is_training=is_training
        )

        with \
        slim.arg_scope([wn.conv2d], **default_conv_args), \
        slim.arg_scope(training_mode_funcs, **training_args):
          
            net = inputs
            assert_shape(net, [None, 32, 32, 3])

            net = tf.cond(normalize_input,
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)
            assert_shape(net, [None, 32, 32, 3])

            net = nn.flip_randomly(net,
                                   horizontally=flip_horizontally,
                                   vertically=False,
                                   name='random_flip')
            net = tf.cond(translate,
                          lambda: nn.random_translate(net, scale=2, name='random_translate'),
                          lambda: net)
            net = nn.gaussian_noise(net, scale=input_noise, name='gaussian_noise')

            net = wn.conv2d(net, 128, scope="conv_1_1")
            net = wn.conv2d(net, 128, scope="conv_1_2")
            net = wn.conv2d(net, 128, scope="conv_1_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_1')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_1')
            assert_shape(net, [None, 16, 16, 128])

            net = wn.conv2d(net, 256, scope="conv_2_1")
            net = wn.conv2d(net, 256, scope="conv_2_2")
            net = wn.conv2d(net, 256, scope="conv_2_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_2')
            net = slim.dropout(net, 1 - dropout_probability, scope='dropout_probability_2')
            assert_shape(net, [None, 8, 8, 256])

            net = wn.conv2d(net, 512, padding='VALID', scope="conv_3_1")
            assert_shape(net, [None, 6, 6, 512])
            net = wn.conv2d(net, 256, kernel_size=[1, 1], scope="conv_3_2")
            net = wn.conv2d(net, 128, kernel_size=[1, 1], scope="conv_3_3")
            net = slim.avg_pool2d(net, [6, 6], scope='avg_pool')
            assert_shape(net, [None, 1, 1, 128])

            net = slim.flatten(net)
            assert_shape(net, [None, 128])

            primary_logits = wn.fully_connected(net, 10, init=is_initialization)
            secondary_logits = wn.fully_connected(net, 10, init=is_initialization)

            with tf.control_dependencies([tf.assert_greater_equal(num_logits, 1),
                                          tf.assert_less_equal(num_logits, 2)]):
                secondary_logits = tf.case([
                    (tf.equal(num_logits, 1), lambda: primary_logits),
                    (tf.equal(num_logits, 2), lambda: secondary_logits),
                ], exclusive=True, default=lambda: primary_logits)

            assert_shape(primary_logits, [None, 10])
            assert_shape(secondary_logits, [None, 10])
            return primary_logits, secondary_logits


def errors(logits, labels, name=None):

    with tf.name_scope(name, "errors") as scope:
        applicable = tf.not_equal(labels, -1)
        labels = tf.boolean_mask(labels, applicable)
        logits = tf.boolean_mask(logits, applicable)
        predictions = tf.argmax(logits, -1)
        labels = tf.cast(labels, tf.int64)
        per_sample = tf.to_float(tf.not_equal(predictions, labels))
        mean = tf.reduce_mean(per_sample, name=scope)
        return mean, per_sample


def classification_costs(logits, labels, name=None):

    with tf.name_scope(name, "classification_costs") as scope:
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count = tf.to_float(tf.shape(per_sample)[0])
        mean = tf.div(labeled_sum, total_count, name=scope)

        return mean, per_sample


def consistency_costs(logits1, logits2, cons_coefficient, mask, consistency_trust, name=None):

    with tf.name_scope(name, "consistency_costs") as scope:
        num_classes = 10
        assert_shape(logits1, [None, num_classes])
        assert_shape(logits2, [None, num_classes])
        assert_shape(cons_coefficient, [])
        softmax1 = tf.nn.softmax(logits1)
        softmax2 = tf.nn.softmax(logits2)

        kl_cost_multiplier = 2 * (1 - 1/num_classes) / num_classes**2 / consistency_trust**2

        def pure_mse():
            costs = tf.reduce_mean((softmax1 - softmax2) ** 2, -1)
            return costs

        def pure_kl():
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=softmax2)
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=softmax2)
            costs = cross_entropy - entropy
            costs = costs * kl_cost_multiplier
            return costs

        def mixture_kl():
            with tf.control_dependencies([tf.assert_greater(consistency_trust, 0.0),
                                          tf.assert_less(consistency_trust, 1.0)]):
                uniform = tf.constant(1 / num_classes, shape=[num_classes])
                mixed_softmax1 = consistency_trust * softmax1 + (1 - consistency_trust) * uniform
                mixed_softmax2 = consistency_trust * softmax2 + (1 - consistency_trust) * uniform
                costs = tf.reduce_sum(mixed_softmax2 * tf.log(mixed_softmax2 / mixed_softmax1), axis=1)
                costs = costs * kl_cost_multiplier
                return costs

        costs = tf.case([
            (tf.equal(consistency_trust, 0.0), pure_mse),
            (tf.equal(consistency_trust, 1.0), pure_kl)
        ], default=mixture_kl)

        costs = costs * tf.to_float(mask) * cons_coefficient
        mean_cost = tf.reduce_mean(costs, name=scope)
        assert_shape(costs, [None])
        assert_shape(mean_cost, [])
        return mean_cost, costs


def total_costs(*all_costs, name=None):
    with tf.name_scope(name, "total_costs") as scope:
        for cost in all_costs:
            assert_shape(cost, [None])
        costs = tf.reduce_sum(all_costs, axis=1)
        mean_cost = tf.reduce_mean(costs, name=scope)
        return mean_cost, costs

```

## Dual Students
- mean teacher 모델은 학습 후반부로 갈 수록 teacher 모델의 가중치가 student 모델의 가중치가 수렴
- 단일 student 모델로 인한 예측 편향 및 불안정성이 있습니다.
- 이를 방지하기 위하여 dual student 모델은 각각 학습되는 student 모델을 2가지 사용합니다.
- 학습되는 모델 중 좀 더 stable하고 안정적인 값이 최종 학습(student)로서 선정됩니다.

### Dual Students의 구조
![image](https://user-images.githubusercontent.com/68594529/209672198-8a1b1161-efce-433d-8ab9-9195aa587fcd.png)

- 해당 구조의 앞단을 보면 augmentation이 2쌍 2번 적용됨을 알 수 있습니다. 또한 노이즈 역시 2번 적용됩니다.
- 이 후 stablization constraint, consistency constraint을 거쳐 예측 결과도 우수한 student가 선정되는 구조를 갖고 있습니다.

### Dual Student 실행 코드
```python
def create_data_loaders(train_transformation, eval_transformation, datadir, args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)
    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)
    ds_size = len(dataset.imgs)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:

        # domain adaptation dataset
        if args.target_domain is not None:
            LOG.info('\nYou set target domain: {0} on script.\n'
                     'This is a domain adaptation experiment.\n'.format(args.target_domain))

            target_dataset_config = datasets.__dict__[args.target_domain]()

            if args.target_domain == 'mnist':
                valid_sources = ['usps']
                if not args.dataset in valid_sources:
                    LOG.error('\nYou set \'mnist\' as the target domain. \n'
                              'However, you use the source domain: \'{0}\'.\n'
                              'The source domain should be \'{1}\''.format(args.dataset, valid_sources))
                target_traindir = '{0}/train'.format(target_dataset_config['datadir'])
                evaldir = '{0}/test'.format(target_dataset_config['datadir'])
                eval_transformation = target_dataset_config['eval_transformation']
            else:
                LOG.error('Unsupport target domain: {0}.\n'.format(args.target_domain))
                
            target_dataset = torchvision.datasets.ImageFolder(target_traindir, target_dataset_config['train_transformation'])
            target_labeled_idxs, target_unlabeled_idxs = data.relabel_dataset(target_dataset, {})

            dataset = ConcatDataset([dataset, target_dataset])
            unlabeled_idxs += [ds_size + i for i in range(0, len(target_dataset.imgs))]

        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader


def create_model(name, num_classes, ema=False):
    LOG.info('=> creating {pretrained} {name} model: {arch}'.format(
        pretrained='pre-trained' if args.pretrained else 'non-pre-trained',
        name=name,
        arch=args.arch))

    model_factory = architectures.__dict__[args.arch]
    model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).cuda()

    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # decline lr
    lr *= ramps.zero_cosine_rampdown(epoch, args.epochs)

    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr


def validate(eval_loader, model, log, global_step, epoch):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = torch.autograd.Variable(inputs, volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        output1, output2 = model(input_var)
        # softmax1, softmax2 = F.softmax(output1, dim=1), F.softmax(output2, dim=1)
        class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        prec = mt_func.accuracy(output1.data, target_var.data, topk=(1, 5))
        prec1, prec5 = prec[0], prec[1]

        meters.update('class_loss', class_loss.data[0], labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info('Test: [{0}/{1}]\t'
                     'Time {meters[batch_time]:.3f}\t'
                     'Data {meters[data_time]:.3f}\t'
                     'Class {meters[class_loss]:.4f}\t'
                     'Prec@1 {meters[top1]:.3f}\t'
                     'Prec@5 {meters[top5]:.3f}'.format(
                         i, len(eval_loader), meters=meters))

    LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'.format(
             top1=meters['top1'], top5=meters['top5']))
    log.record(epoch, {'step': global_step, **meters.values(),
               **meters.averages(), **meters.sums()})

    return meters['top1'].avg


def train_epoch(train_loader, l_model, r_model, l_optimizer, r_optimizer, epoch, log):
    global global_step

    meters = AverageMeterSet()

    # define criterions
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    residual_logit_criterion = losses.symmetric_mse_loss
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
        stabilization_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
        stabilization_criterion = losses.softmax_kl_loss

    l_model.train()
    r_model.train()

    end = time.time()
    for i, ((l_input, r_input), target) in enumerate(train_loader):
        meters.update('data_time', time.time() - end)

        # adjust learning rate
        adjust_learning_rate(l_optimizer, epoch, i, len(train_loader))
        adjust_learning_rate(r_optimizer, epoch, i, len(train_loader))
        meters.update('l_lr', l_optimizer.param_groups[0]['lr'])
        meters.update('r_lr', r_optimizer.param_groups[0]['lr'])

        # prepare data
        l_input_var = Variable(l_input)
        r_input_var = Variable(r_input)
        le_input_var = Variable(r_input, requires_grad=False, volatile=True)
        re_input_var = Variable(l_input, requires_grad=False, volatile=True)
        target_var = Variable(target.cuda(async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        unlabeled_minibatch_size = minibatch_size - labeled_minibatch_size
        assert labeled_minibatch_size >= 0 and unlabeled_minibatch_size >= 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)
        meters.update('unlabeled_minibatch_size', unlabeled_minibatch_size)

        # forward
        l_model_out = l_model(l_input_var)
        r_model_out = r_model(r_input_var)
        le_model_out = l_model(le_input_var)
        re_model_out = r_model(re_input_var)

        if isinstance(l_model_out, Variable):
            assert args.logit_distance_cost < 0
            l_logit1 = l_model_out
            r_logit1 = r_model_out
            le_logit1 = le_model_out
            re_logit1 = re_model_out
        elif len(l_model_out) == 2:
            assert len(r_model_out) == 2
            l_logit1, l_logit2 = l_model_out
            r_logit1, r_logit2 = r_model_out
            le_logit1, le_logit2 = le_model_out
            re_logit1, re_logit2 = re_model_out

        # logit distance loss from mean teacher
        if args.logit_distance_cost >= 0:
            l_class_logit, l_cons_logit = l_logit1, l_logit2
            r_class_logit, r_cons_logit = r_logit1, r_logit2
            le_class_logit, le_cons_logit = le_logit1, le_logit2
            re_class_logit, re_cons_logit = re_logit1, re_logit2

            l_res_loss = args.logit_distance_cost * residual_logit_criterion(l_class_logit, l_cons_logit) / minibatch_size
            r_res_loss = args.logit_distance_cost * residual_logit_criterion(r_class_logit, r_cons_logit) / minibatch_size
            meters.update('l_res_loss', l_res_loss.data[0])
            meters.update('r_res_loss', r_res_loss.data[0])
        else:
            l_class_logit, l_cons_logit = l_logit1, l_logit1
            r_class_logit, r_cons_logit = r_logit1, r_logit1
            le_class_logit, le_cons_logit = le_logit1, le_logit1
            re_class_logit, re_cons_logit = re_logit1, re_logit1

            l_res_loss = 0.0
            r_res_loss = 0.0
            meters.update('l_res_loss', 0.0)
            meters.update('r_res_loss', 0.0)

        # classification loss
        l_class_loss = class_criterion(l_class_logit, target_var) / minibatch_size
        r_class_loss = class_criterion(r_class_logit, target_var) / minibatch_size
        meters.update('l_class_loss', l_class_loss.data[0])
        meters.update('r_class_loss', r_class_loss.data[0])

        l_loss, r_loss = l_class_loss, r_class_loss
        l_loss += l_res_loss
        r_loss += r_res_loss

        # consistency loss
        consistency_weight = args.consistency_scale * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

        le_class_logit = Variable(le_class_logit.detach().data, requires_grad=False)
        l_consistency_loss = consistency_weight * consistency_criterion(l_cons_logit, le_class_logit) / minibatch_size
        meters.update('l_cons_loss', l_consistency_loss.data[0])
        l_loss += l_consistency_loss

        re_class_logit = Variable(re_class_logit.detach().data, requires_grad=False)
        r_consistency_loss = consistency_weight * consistency_criterion(r_cons_logit, re_class_logit) / minibatch_size
        meters.update('r_cons_loss', r_consistency_loss.data[0])
        r_loss += r_consistency_loss

        # stabilization loss
        # value (cls_v) and index (cls_i) of the max probability in the prediction
        l_cls_v, l_cls_i = torch.max(F.softmax(l_class_logit, dim=1), dim=1)
        r_cls_v, r_cls_i = torch.max(F.softmax(r_class_logit, dim=1), dim=1)
        le_cls_v, le_cls_i = torch.max(F.softmax(le_class_logit, dim=1), dim=1)
        re_cls_v, re_cls_i = torch.max(F.softmax(re_class_logit, dim=1), dim=1)

        l_cls_i = l_cls_i.data.cpu().numpy()
        r_cls_i = r_cls_i.data.cpu().numpy()
        le_cls_i = le_cls_i.data.cpu().numpy()
        re_cls_i = re_cls_i.data.cpu().numpy()

        # stable prediction mask 
        l_mask = (l_cls_v > args.stable_threshold).data.cpu().numpy()
        r_mask = (r_cls_v > args.stable_threshold).data.cpu().numpy()
        le_mask = (le_cls_v > args.stable_threshold).data.cpu().numpy()
        re_mask = (re_cls_v > args.stable_threshold).data.cpu().numpy()

        # detach logit -> for generating stablilization target 
        in_r_cons_logit = Variable(r_cons_logit.detach().data, requires_grad=False)
        tar_l_class_logit = Variable(l_class_logit.clone().detach().data, requires_grad=False)

        in_l_cons_logit = Variable(l_cons_logit.detach().data, requires_grad=False)
        tar_r_class_logit = Variable(r_class_logit.clone().detach().data, requires_grad=False)

        # generate target for each sample
        for sdx in range(0, minibatch_size):
            l_stable = False
            if l_mask[sdx] == 0 and le_mask[sdx] == 0:              
                # unstable: do not satisfy the 2nd condition
                tar_l_class_logit[sdx, ...] = in_r_cons_logit[sdx, ...]
            elif l_cls_i[sdx] != le_cls_i[sdx]:
                # unstable: do not satisfy the 1st condition
                tar_l_class_logit[sdx, ...] = in_r_cons_logit[sdx, ...]
            else:
                l_stable = True

            r_stable = False
            if r_mask[sdx] == 0 and re_mask[sdx] == 0:
                # unstable: do not satisfy the 2nd condition
                tar_r_class_logit[sdx, ...] = in_l_cons_logit[sdx, ...]
            elif r_cls_i[sdx] != re_cls_i[sdx]:
                # unstable: do not satisfy the 1st condition
                tar_r_class_logit[sdx, ...] = in_l_cons_logit[sdx, ...]
            else:
                r_stable = True

            # calculate stability if both models are stable for a sample
            if l_stable and r_stable:
                # compare by consistency
                l_sample_cons = consistency_criterion(l_cons_logit[sdx:sdx+1, ...], le_class_logit[sdx:sdx+1, ...])
                r_sample_cons = consistency_criterion(r_cons_logit[sdx:sdx+1, ...], re_class_logit[sdx:sdx+1, ...])
                if l_sample_cons.data.cpu().numpy()[0] < r_sample_cons.data.cpu().numpy()[0]:
                    # loss: l -> r
                    tar_r_class_logit[sdx, ...] = in_l_cons_logit[sdx, ...]
                elif l_sample_cons.data.cpu().numpy()[0] > r_sample_cons.data.cpu().numpy()[0]:
                    # loss: r -> l
                    tar_l_class_logit[sdx, ...] = in_r_cons_logit[sdx, ...]

        # calculate stablization weight
        stabilization_weight = args.stabilization_scale * ramps.sigmoid_rampup(epoch, args.stabilization_rampup)
        if not args.exclude_unlabeled:
            stabilization_weight = (unlabeled_minibatch_size / minibatch_size) * stabilization_weight

        # stabilization loss for r model
        if args.exclude_unlabeled:
            r_stabilization_loss = stabilization_weight * stabilization_criterion(r_cons_logit, tar_l_class_logit) / minibatch_size
        else:
            for idx in range(unlabeled_minibatch_size, minibatch_size):
                tar_l_class_logit[idx, ...] = in_r_cons_logit[idx, ...]
            r_stabilization_loss = stabilization_weight * stabilization_criterion(r_cons_logit, tar_l_class_logit) / unlabeled_minibatch_size
        meters.update('r_stable_loss', r_stabilization_loss.data[0])
        r_loss += r_stabilization_loss

        # stabilization loss for l model
        if args.exclude_unlabeled:
            l_stabilization_loss = stabilization_weight * stabilization_criterion(l_cons_logit, tar_r_class_logit) / minibatch_size
        else:
            for idx in range(unlabeled_minibatch_size, minibatch_size):
                tar_r_class_logit[idx, ...] = in_l_cons_logit[idx, ...]
            l_stabilization_loss = stabilization_weight * stabilization_criterion(l_cons_logit, tar_r_class_logit) / unlabeled_minibatch_size

        meters.update('l_stable_loss', l_stabilization_loss.data[0])
        l_loss += l_stabilization_loss

        if np.isnan(l_loss.data[0]) or np.isnan(r_loss.data[0]):
            LOG.info('Loss value equals to NAN!')
            continue
        assert not (l_loss.data[0] > 1e5), 'L-Loss explosion: {}'.format(l_loss.data[0])
        assert not (r_loss.data[0] > 1e5), 'R-Loss explosion: {}'.format(r_loss.data[0])
        meters.update('l_loss', l_loss.data[0])
        meters.update('r_loss', r_loss.data[0])

        # calculate prec and error
        l_prec = mt_func.accuracy(l_class_logit.data, target_var.data, topk=(1, ))[0]
        r_prec = mt_func.accuracy(r_class_logit.data, target_var.data, topk=(1, ))[0]

        meters.update('l_top1', l_prec[0], labeled_minibatch_size)
        meters.update('l_error1', 100. - l_prec[0], labeled_minibatch_size)

        meters.update('r_top1', r_prec[0], labeled_minibatch_size)
        meters.update('r_error1', 100. - r_prec[0], labeled_minibatch_size)

        # update model
        l_optimizer.zero_grad()
        l_loss.backward()
        l_optimizer.step()

        r_optimizer.zero_grad()
        r_loss.backward()
        r_optimizer.step()

        # record
        global_step += 1
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info('Epoch: [{0}][{1}/{2}]\t'
                     'Batch-T {meters[batch_time]:.3f}\t'
                     'L-Class {meters[l_class_loss]:.4f}\t'
                     'R-Class {meters[r_class_loss]:.4f}\t'
                     'L-Res {meters[l_res_loss]:.4f}\t'
                     'R-Res {meters[r_res_loss]:.4f}\t'
                     'L-Cons {meters[l_cons_loss]:.4f}\t'
                     'R-Cons {meters[r_cons_loss]:.4f}\n'
                     'L-Stable {meters[l_stable_loss]:.4f}\t'
                     'R-Stable {meters[r_stable_loss]:.4f}\t'
                     'L-Prec@1 {meters[l_top1]:.3f}\t'
                     'R-Prec@1 {meters[r_top1]:.3f}\t'
                     .format(epoch, i, len(train_loader), meters=meters))

            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()})

def main(context):
    global best_prec1
    global global_step

    # create loggers
    checkpoint_path = context.transient_dir
    training_log = context.create_train_log('training')
    l_validation_log = context.create_train_log('l_validation')
    r_validation_log = context.create_train_log('r_validation')

    # create dataloaders
    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)

    # create models
    l_model = create_model(name='l', num_classes=num_classes)
    r_model = create_model(name='r', num_classes=num_classes)
    LOG.info(parameters_string(l_model))
    LOG.info(parameters_string(r_model))

    # create optimizers
    l_optimizer = torch.optim.SGD(params=l_model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)
    r_optimizer = torch.optim.SGD(params=r_model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)

    # restore saved checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '=> no checkpoint found at: {}'.format(args.resume)
        LOG.info('=> loading checkpoint: {}'.format(args.resume))

        checkpoint = torch.load(args.resume)

        # globel parameters
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        
        # models and optimizers
        l_model.load_state_dict(checkpoint['l_model'])
        r_model.load_state_dict(checkpoint['r_model'])
        l_optimizer.load_state_dict(checkpoint['l_optimizer'])
        r_optimizer.load_state_dict(checkpoint['r_optimizer'])

        LOG.info('=> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    # validation
    if args.validation:
        LOG.info('Validating the left model: ')
        validate(eval_loader, l_model, l_validation_log, global_step, args.start_epoch)
        LOG.info('Validating the right model: ')
        validate(eval_loader, r_model, r_validation_log, global_step, args.start_epoch)
        return

    # training
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        train_epoch(train_loader, l_model, r_model, l_optimizer, r_optimizer, epoch, training_log)
        LOG.info('--- training epoch in {} seconds ---'.format(time.time() - start_time))

        is_best = False
        if args.validation_epochs and (epoch + 1) % args.validation_epochs == 0:
            start_time = time.time()

            LOG.info('Validating the left model: ')
            l_prec1 = validate(eval_loader, l_model, l_validation_log, global_step, epoch + 1)
            LOG.info('Validating the right model: ')
            r_prec1 = validate(eval_loader, r_model, r_validation_log, global_step, epoch + 1)

            LOG.info('--- validation in {} seconds ---'.format(time.time() - start_time))
            better_prec1 = l_prec1 if l_prec1 > r_prec1 else r_prec1
            best_prec1 = max(better_prec1, best_prec1)
            is_best = better_prec1 > best_prec1

        # save checkpoint
        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            mt_func.save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'best_prec1': best_prec1,
                'arch': args.arch,
                'l_model': l_model.state_dict(),
                'r_model': r_model.state_dict(),
                'l_optimizer':l_optimizer.state_dict(),
                'r_optimizer':r_optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)

    LOG.info('Best top1 prediction: {0}'.format(best_prec1))
---

## Conclusion
- 5가지 모델들의 구조 및 실행코드들을 살펴봄

## References
[고려대학교 강필성 교수님](https://github.com/pilsung-kang)<br>
[Ladder-network](https://github.com/divamgupta/ladder_network_keras)<br>
[PI-Model](https://github.com/geosada/PI/blob/master/PI.py)<br>
[Temporal-Ensemble](https://github.com/ferretj/temporal-ensembling)<br>
[Mean-teacher](https://github.com/CuriousAI/mean-teacher)<br>
[Dual-student](https://github.com/ZHKKKe/DualStudent)<br>
