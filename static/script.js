// Wait for the DOM to fully load
document.addEventListener('DOMContentLoaded', () => {
  
  // Functionality for the "Choose file" button
  const uploadButton = document.querySelector('.upload-button');
  const fileInput = document.querySelector('input[type="file"]');
  const outputImage = document.getElementById('output-image');
  const resultText = document.getElementById('result-text');



  // const fileInput = document.querySelector('#file-upload');

  if (uploadButton && fileInput) {
    uploadButton.addEventListener('click', () => {
      fileInput.click(); // Trigger the file input click
    });
    
    // Update button text when a file is selected
    fileInput.addEventListener('change', (event) => {
      const fileName = event.target.files[0]?.name || 'No file chosen';
      uploadButton.innerText = fileName;

      // Start the upload progress animation
      startUpload();
    

    const file = event.dataTransfer.files[0];
    fileInput.files = event.dataTransfer.files;  
    detectTumor(file);

    });
  }


  // Functionality for "Start over" button
  const startOverButton = document.querySelector('.start-over-button');
  if (startOverButton) {
    // startOverButton.addEventListener('click', () => {
    //   window.location.href="{{ url_for('detect') }}"; // Redirect to detect page
    // });

    startOverButton.addEventListener('click', resetUpload);
  }
});

// Function for starting the upload
function startUpload() {
  const progressBar = document.getElementById('progress-bar');
  let progress = 0;
  

  const interval = setInterval(() => {
    if (progress >= 100) {
      clearInterval(interval);
      
    } else {
      progress += 10; // Increment progress by 10% each time
      progressBar.style.width = progress + '%';
    }
  }, 300); // Adjust the speed of the progress (300ms for a smooth animation)  
}



function resetUpload() {
  const progressBar = document.getElementById('progress-bar');
  progressBar.style.width = '0%';
  document.getElementById('result-text');
  
  // Reset the upload button text
  const uploadButton = document.querySelector('.upload-button');
  uploadButton.innerText = 'Choose file'; // Reset to default text

  // Clear the file input
  document.getElementById('file-upload').value = null; // Clear the file input
}

 
// Function to detect tumors
function detectTumor(file) {
  const reader = new FileReader();
  reader.onload = function(event) {
    const imageData = event.target.result;

    reader.addEventListener('progress', (progressEvent) => {
      const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
      const progressBar = document.getElementById('progress-bar');
      progressBar.style.width = `${progress}%`;
    });

    // Load TensorFlow.js model
    const model = tf.loadLayersModel('vgg19_model2.keras');
    model.predict(imageData).then(predictions => {
      // Display the output image
      // Update output image with the detected tumor
      const tumorPresent = predictions[0][0].toFixed(2);
      const predictedClass = predictions[0][1].toFixed(2);
      const tumorLocation = predictions[0][2].toFixed(2);


      // Send the prediction data to the Flask server
      fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/keras'
        },

        body: JSON.stringify({
          'tumor_present': tumorPresent,
          'predicted_class': predictedClass,
          'tumor_location': tumorLocation
        })
      })

      .then(response => response.keras())
      .then(data => {
        // Display the result
        outputImage.src = `data:image/jpeg;base64,${data['img_data']}`;
      });
    });
  };
  reader.readAsDataURL(file);
}


