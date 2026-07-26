// main app js for xray analysis page

document.getElementById('xrayFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(ev) {
            document.getElementById('imagePreview').src = ev.target.result;
            document.getElementById('previewArea').classList.remove('d-none');
        };
        reader.readAsDataURL(file);
    }
});

// main app js for xray analysis page

document.getElementById('xrayFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(ev) {
            document.getElementById('imagePreview').src = ev.target.result;
            document.getElementById('previewArea').classList.remove('d-none');
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('xrayFile');
    if (!fileInput.files[0]) {
        alert('Please select an image first');
        return;
    }
    await runAnalysis(fileInput.files[0]);
});

async function analyzeSample(imageUrl, filename) {
    try {
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });

        // show preview
        document.getElementById('imagePreview').src = imageUrl;
        document.getElementById('previewArea').classList.remove('d-none');

        await runAnalysis(file);
    } catch (err) {
        alert('Could not load sample image: ' + err.message);
    }
}

async function runAnalysis(file) {
    const notes = document.getElementById('patientNotes').value;

    document.getElementById('loadingSpinner').classList.remove('d-none');
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('resultsPanel').classList.add('d-none');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('patient_notes', notes);

    try {
        const response = await fetch('/api/v1/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Something went wrong');
        }

        const data = await response.json();
        showResults(data);

    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        document.getElementById('loadingSpinner').classList.add('d-none');
        document.getElementById('analyzeBtn').disabled = false;
    }
}

function showResults(data) {
    document.getElementById('resultsPanel').classList.remove('d-none');

    const predEl = document.getElementById('predClass');
    predEl.textContent = data.predicted_class;
    predEl.className = data.predicted_class === 'Pneumonia' ? 'pred-pneumonia' : 'pred-normal';

    document.getElementById('predConf').textContent = data.confidence;

    // update progress bars
    const normalProb = data.probabilities.Normal || 0;
    const pneumoniaProb = data.probabilities.Pneumonia || 0;
    document.getElementById('probNormal').style.width = normalProb + '%';
    document.getElementById('probNormal').textContent = 'Normal ' + normalProb + '%';
    document.getElementById('probPneumonia').style.width = pneumoniaProb + '%';
    document.getElementById('probPneumonia').textContent = 'Pneumonia ' + pneumoniaProb + '%';

    // gradcam
    if (data.gradcam_image) {
        document.getElementById('gradcamImg').src = 'data:image/png;base64,' + data.gradcam_image;
    }

    // report
    document.getElementById('reportText').textContent = data.report;
}
