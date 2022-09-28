# Loading Libraries
library(tidyverse)
library(keras)
library(tensorflow)
#install_keras()
#install_tensorflow()

# Constants
img_rows <- 28
img_cols <- 28
n_classes <- 10

# Loading Data
test <- read_rds("test.rds")
label <- read_rds("label_test.rds")
train <- read_rds("train.rds")


x_train <- train %>%
  filter(Flag == 0) %>%
  select(-label, -Flag) %>%
  mutate_all(function(x) x/255) %>%
  as.matrix()


y_train <- train %>%
  filter(Flag == 0) %>%
  select(label) %>% as.matrix() %>%
  to_categorical(num_classes = 10)


# CNN

# Redefine dimension of train/test inputs
x_train_cnn <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
cnn_test <- array_reshape(test, c(nrow(test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# We can see a example of a image number in the obect x_train_cnn
image(x_train_cnn[5,,,1])

model_cnn <- keras_model_sequential() %>%
  layer_conv_2d(
    filters = 32,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = input_shape
  ) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  # these are the embeddings (activations) we are going to visualize
  layer_dense(units = 128, activation = 'relu', name = 'features') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile model
model_cnn %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

summary(model_cnn)

history <- model_cnn %>% fit(
  x_train_cnn, y_train,
  epochs = 15, batch_size = 128,
  validation_split = 0.2,
  callbacks = list(
    callback_early_stopping(monitor = "val_acc", min_delta = 0.01,
                            patience =8,
                            verbose=0,
                            mode = "auto",
                            restore_best_weights = TRUE) )
)


# Evaluate
model_cnn %>% evaluate(cnn_test, label)


#Another strategy is feeding the neural neetwork with more variables and observations, making a resample 
# for example. Lets try

#new dataset create by boostrap
newdata<- train %>% 
  filter(Flag == 0) %>%
  sample_frac(1, replace = TRUE)

dim(newdata)
#Uniting the old dataset to the "newdata" created and padronizing after

x_train_augmented<-train %>% 
  filter(Flag == 0) %>%
  bind_rows(newdata) %>% 
  select(-label, -Flag) %>% 
  mutate_all(function(x) x/255) %>%
  as.matrix()

dim(x_train_augmented)

#Just a labels of the training dataset with a Uniting "newdata"
y_train_augmented <- train %>%
  filter(Flag == 0) %>%
  bind_rows(newdata) %>% 
  select(label) %>%
  as.matrix() %>%
  to_categorical(num_classes = 10)



# Redefine dimension of train/test inputs
x_train_cnn <- array_reshape(x_train_augmented, c(nrow(x_train_augmented), img_rows, img_cols, 1))
cnn_test <- array_reshape(test, c(nrow(test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)


model_cnn <- keras_model_sequential() %>%
  layer_conv_2d(
    filters = 64,
    kernel_size = c(3, 3),
    activation = 'relu',
    input_shape = input_shape
  ) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3, 3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  # these are the embeddings (activations) we are going to visualize
  layer_dense(units = 1024, activation = 'relu', name = 'features') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = "softmax")

# Compile model
model_cnn %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)


history <- model_cnn %>% fit( x_train_cnn, y_train_augmented,
                              epochs = 15, batch_size = 128,
                              validation_split = 0.2,
                              callbacks = list(
                                callback_early_stopping(monitor = "val_acc", min_delta = 0.01,
                                                        patience =8,
                                                        verbose=0,
                                                        mode = "auto",
                                                        restore_best_weights = TRUE) )
)

# Evaluate
model_cnn %>% evaluate(cnn_test, label)

test2<-test %>%  as.matrix()
a <- predict_classes(model_cnn, cnn_test) %>%  as.matrix()
results<- a %>% as_tibble()
dim(results)
colnames(results)<-list(c("Label"))

utils::write.csv2(results, file="results.csv", col.names = "Label")


