import tensorflow as tf

tall = 170
shoes = 260

# shoes = tall * a + b
a = tf.Variable(0.1)
b = tf.Variable(0.2)
#손실함수
def loss_function():
    predict_value = tall * a + b
    return tf.square(260 - predict_value)



#경사 하강법
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
    opt.minimize(loss_function, var_list=[a, b])
    print(a.numpy(),b.numpy())


