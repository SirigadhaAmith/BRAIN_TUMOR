from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mail import Mail, Message
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose
import numpy as np
import cv2, os
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the pre-trained model
model = load_model('vgg19_model2.keras')

print('Model loaded. Check http://127.0.0.1:5000')

# # Ensure the 'uploads' directory exists
# if not os.path.exists('uploads'):
#     os.makedirs('uploads')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

# @app.route('/login', methods=['POST'])
# def login():
#     # Get form data
#     email = request.form['email']
#     password = request.form['password']
    
#     # Redirect to the home page after login
#     return redirect(url_for('home'))

@app.route('/home', methods=['GET'])
def home():
    # Home page after login
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    # Home page after login
    return render_template('about.html')

@app.route('/detect', methods=['GET'])
def detect():
    # Home page after login
    return render_template('detect.html')

# @app.route('/contact', methods=['GET'])
# def contact():
#     # Home page after login
#     return render_template('contact.html')

# Function to predict tumor presence and type
def predict_tumor(img_path):
    # Read the image
    img = cv2.imread(img_path)
    print("Image read successfully")

    # Denoising using Gaussian filter
    denoised_img = cv2.GaussianBlur(img, (5, 5), 0)
    print("Gaussian blur applied")

    # Median filter for smoothing
    blured_image = cv2.medianBlur(denoised_img, 1)
    print("Median blur applied")

    # Initial mean ti
    ti = 75.41121758241758

    # Get top, bottom, left, and right pixels
    top = next(((i, j) for i in range(blured_image.shape[0]) for j in range(blured_image.shape[1]) if np.mean(blured_image[i, j]) > ti), (0, 0))
    bottom = next(((i, j) for i in range(blured_image.shape[0] - 1, -1, -1) for j in range(blured_image.shape[1]) if np.mean(blured_image[i, j]) > ti), (blured_image.shape[0] - 1, blured_image.shape[1] - 1))
    left = next(((i, j) for j in range(blured_image.shape[1]) for i in range(blured_image.shape[0]) if np.mean(blured_image[i, j]) > ti), (0, 0))
    right = next(((i, j) for j in range(blured_image.shape[1] - 1, -1, -1) for i in range(blured_image.shape[0]) if np.mean(blured_image[i, j]) > ti), (blured_image.shape[0] - 1, blured_image.shape[1] - 1))

    # Cropped image
    cropped_image = blured_image[top[0]:bottom[0], left[1]:right[1]]


    # Resize image for prediction
    brain_image_resized = cv2.resize(cropped_image, (224, 224))  # Assuming the model expects 224x224 input size

    # Normalize image for prediction
    brain_image_normalized = brain_image_resized / 255.0

    # Expand dimensions to match the input shape of the model
    brain_image_input = np.expand_dims(brain_image_normalized, axis=0)

    # Make prediction
    prediction = model.predict(brain_image_input)
    print("Prediction made")

    # Decode prediction
    class_labels = ["glioma", "meningioma", "notumor", "pituitary"]
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]

    # Determine tumor presence
    tumor_present = "Yes" if predicted_class != "notumor" else "No"

    return tumor_present, predicted_class, cropped_image

# PGGAN Model
class PGGAN:
    def __init__(self, img_rows=32, img_cols=32, channels=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.latent_dim = 100

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes the noise vector as input and generates the output image
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model, we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer='adam')

    def build_generator(self):
        model = Sequential()
        # Foundation for 4x4 image
        n_nodes = 256 * 4 * 4
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Reshape((4, 4, 256)))
        # Upsample to 8x8
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Upsample to 16x16
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Upsample to 32x32
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Upsample to 64x64
        model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Upsample to 128x128
        model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Upsample to 256x256
        model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Output layer
        model.add(Conv2D(self.channels, (3, 3), activation='tanh', padding='same'))
        return model

    def build_discriminator(self):
        model = Sequential()
        # Downsample to 128x128
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=(256, 256, self.channels)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Downsample to 64x64
        model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Downsample to 32x32
        model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Downsample to 16x16
        model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Downsample to 8x8
        model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Downsample to 4x4
        model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        # Classifier
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def train(self, X_train, epochs, batch_size=128, sample_interval=50):
        # Rescale -1 to 1
        X_train = (X_train - 127.5) / 127.5
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]
            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            valid_y = np.array([1] * batch_size)
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Print the progress
            if epoch % sample_interval == 0:
                print(f"{epoch} [D loss: {d_loss[0]} | acc.: {100*d_loss[1]}] [G loss: {g_loss}]")
    
def generate_synthetic_images(num_images):
    gan = PGGAN(img_rows=256, img_cols=256, channels=3)
    noise = np.random.normal(0, 1, (num_images, gan.latent_dim))
    synthetic_images = gan.generator.predict(noise)
    return synthetic_images


@app.route('/predict', methods=['POST'])
def upload_file():
       # Check if a file was uploaded
    if 'image' not in request.files:
        return "Error: No file part"
    
    if request.method == 'POST':
        file = request.files['image']

        # Check if a file is selected
        if file.filename == '':
            return "Error: No selected file"
    
            # Save the file and predict the result
        if file:
            img_path = os.path.join('uploads', file.filename)            
            file.save(img_path)
            tumor_present, predicted_class, brain_image = predict_tumor(img_path)

            if predicted_class is not None:
                result = {
                    "tumor_present": tumor_present,
                    "predicted_class": predicted_class
                }
        

                # Convert image to 0-255 range for display
                display_image = np.uint8(brain_image)

                # Resize the image to a smaller size
                smaller_image = cv2.resize(display_image, (280, 280))  # Resize to smaller size

                # Apply colormap to the image
                colormap = plt.cm.viridis
                heatmap_image = colormap(smaller_image[:, :, 0] / 255.0)

                # Convert colormap image to uint8
                heatmap_image_uint8 = (heatmap_image[:, :, :3] * 255).astype(np.uint8)

                # Encode the final heatmap image for display in the result.html template
                image_stream = io.BytesIO()
                plt.imsave(image_stream, heatmap_image_uint8, format='png')
                image_stream.seek(0)
                encoded_img = base64.b64encode(image_stream.read()).decode('utf-8')

                # Generate high-resolution image using PGGAN
                gan = PGGAN(img_rows=256, img_cols=256, channels=3)
                noise = np.random.normal(0, 1, (1, gan.latent_dim))
                synthetic_image = gan.generator.predict(noise)
                synthetic_image = 0.5 * synthetic_image + 0.5  # Rescale to [0, 1]
                synthetic_image_uint8 = (synthetic_image[0] * 255).astype(np.uint8)

                # Encode the synthetic image
                synthetic_image_stream = io.BytesIO()
                plt.imsave(synthetic_image_stream, synthetic_image_uint8, format='png')
                synthetic_image_stream.seek(0)
                encoded_synthetic_img = base64.b64encode(synthetic_image_stream.read()).decode('utf-8')

                return render_template('result.html', result=result, img_data=encoded_img, synthetic_img_data=encoded_synthetic_img)
    return "Error: No file uploaded"


# Configure Flask-Mail for Gmail (or any other email service)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'rsworkspace3111@gmail.com'  # Your email address
app.config['MAIL_PASSWORD'] = 'Charan@3111'  # Your email password or app-specific password
app.config['MAIL_DEFAULT_SENDER'] = 'rsworkspace3111@gmail.com'  # Default sender for outgoing emails

mail = Mail(app)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Compose the email
        msg = Message(f"Message from {name}", 
                      recipients=['rsworkspace3111@gmail.com'])  # Change to your receiving email
        
        msg.body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        
        try:
            # Send the email
            mail.send(msg)
            flash('Message sent successfully!', 'success')
            return redirect(url_for('contact'))
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'danger')
            return redirect(url_for('contact'))
    
    return render_template('contact.html')

# if name == 'main':
#     app.secret_key = 'your-secret-key'  # Needed for flash messages
#     app.run(debug=True)


if __name__ == '__main__':
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
    app.run(debug=True)
