
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import random


dataset = os.path.dirname(os.path.abspath(__file__))
train_data = os.path.join(dataset, "train")
test_data = os.path.join(dataset, "test")
output_dir = os.path.join(dataset, "outputs")
os.makedirs(output_dir, exist_ok=True)




IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_data,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_generator.class_indices)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


for layer in base_model.layers[:-10]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)


model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



epoch_checkpoint_path = os.path.join(output_dir, "epoch_{epoch:02d}_valacc_{val_accuracy:.4f}.keras")
epoch_checkpoint = ModelCheckpoint(
    filepath=epoch_checkpoint_path,
    monitor='val_accuracy',
    save_best_only=False,   # Save every epoch
    verbose=1
)


best_loss_checkpoint_path = os.path.join(output_dir, "best_loss_model.keras")
best_loss_checkpoint = ModelCheckpoint(
    filepath=best_loss_checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

callbacks = [epoch_checkpoint, best_loss_checkpoint, early_stop]


EPOCHS = 25

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# we plot training & validation accuracy/loss values
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend(); plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss')

plt.tight_layout()
plot_path = os.path.join(output_dir, "training_curves.png")
plt.savefig(plot_path)
plt.show()
print(f"Saved training curves → {plot_path}")


test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc:.4f}")


Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys(),
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
cm_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()
print(f"Saved confusion matrix → {cm_path}")


print("\nClassification Report:\n")
print(classification_report(test_generator.classes, y_pred, target_names=test_generator.class_indices.keys()))


model.save(os.path.join(output_dir, "final_model.keras"))
model.save_weights(os.path.join(output_dir, "final_model_weights.h5"))

filenames = test_generator.filenames
results_df = pd.DataFrame({
    "Image": filenames,
    "Predicted_Emotion": [list(test_generator.class_indices.keys())[i] for i in y_pred]
})
csv_path = os.path.join(output_dir, "predictions.csv")
results_df.to_csv(csv_path, index=False)
print(f"Saved predictions → {csv_path}")


sample_idx = random.sample(range(len(filenames)), 9)
plt.figure(figsize=(10,10))
for i, idx in enumerate(sample_idx):
    img_path = os.path.join(test_data, filenames[idx])
    img = plt.imread(img_path)
    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(f"Pred: {results_df['Predicted_Emotion'][idx]}")
    plt.axis('off')

samples_path = os.path.join(output_dir, "sample_predictions.png")
plt.savefig(samples_path)
plt.show()
print(f"Saved sample predictions → {samples_path}")
