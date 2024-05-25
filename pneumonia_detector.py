from fastai.vision.all import *

path = Path('chest_xray')

# Load the data
dls = ImageDataLoaders.from_folder(
    path,
    train='train',
    valid='val',
    item_tfms=Resize(224),
    batch_tfms=aug_transforms()
)

# Show a batch of images
dls.show_batch()

# Create a CNN learner with a pre-trained ResNet50 model
learn = cnn_learner(dls, resnet50, metrics=accuracy)

# Train the model
learn.fine_tune(4)

# Evaluate the model
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(15, 10))

# Save the model
learn.save('pneumonia-detection-model')
# Load the saved model
learn = load_learner('pneumonia-detection-model.pkl')

# Make a prediction on a new image
img = PILImage.create('path/to/new/image.jpg')
pred, pred_idx, probs = learn.predict(img)
print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
