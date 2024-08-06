document.getElementById('uploadForm').addEventListener('submit',async function (event){
    event.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const formData = new FormData();
    formData.append('file',fileInput.files[0]);

    try {
        const response = await fetch('https://x-ray-check.onrender.com:8080/predict',{
            method : 'POST',
            body : formData
        });
        const result = await response.json();

        const output = document.getElementById('output');
        if (response.ok){
            output.innerHTML = result;
        } else {
            output.innerHTML = result;
        }
    }catch(error){
        console.error('Error : ', error);
        alert('Error uploading file')
    }
});