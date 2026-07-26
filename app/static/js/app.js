// main app js for xray analysis page

const SAMPLES = {
    normal: { url: '/static/demo/NORMAL/demo_normal_0.jpeg', name: 'normal_sample.jpeg' },
    pneumonia: { url: '/static/demo/PNEUMONIA/demo_pneumonia_0.jpeg', name: 'pneumonia_sample.jpeg' },
};

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('xrayFile').addEventListener('change', onFileSelected);
    document.getElementById('uploadForm').addEventListener('submit', onFormSubmit);
    document.getElementById('tryNormalBtn').addEventListener('click', function() {
        loadSample('normal');
    });
    document.getElementById('tryPneumoniaBtn').addEventListener('click', function() {
        loadSample('pneumonia');
    });
});

function onFileSelected(e) {
    const file = e.target.files[0];
    if (file) {
        showPreview(file);
    }
}

async function onFormSubmit(e) {
    e.preventDefault();
    const fileInput = document.getElementById('xrayFile');
    if (!fileInput.files[0]) {
        alert('Please select an image first');
        return;
    }
    await runAnalysis(fileInput.files[0]);
}

function showPreview(fileOrUrl) {
    const preview = document.getElementById('imagePreview');
    if (fileOrUrl instanceof File) {
        preview.src = URL.createObjectURL(fileOrUrl);
    } else {
        preview.src = fileOrUrl;
    }
    document.getElementById('previewArea').classList.remove('d-none');
}

function setFileInput(file) {
    const fileInput = document.getElementById('xrayFile');
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
}

async function loadSample(type) {
    const sample = SAMPLES[type];
    const btn = type === 'normal' ? document.getElementById('tryNormalBtn') : document.getElementById('tryPneumoniaBtn');

    try {
        btn.disabled = true;
        btn.textContent = 'Loading...';

        const response = await fetch(sample.url);
        if (!response.ok) {
            throw new Error('Sample image not found on server');
        }

        const blob = await response.blob();
        const file = new File([blob], sample.name, { type: blob.type || 'image/jpeg' });

        setFileInput(file);
        showPreview(sample.url);
        await runAnalysis(file);

    } catch (err) {
        alert('Could not load sample: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = type === 'normal' ? 'Try Normal' : 'Try Pneumonia';
    }
}

async function runAnalysis(file) {
    const notes = document.getElementById('patientNotes').value;

    document.getElementById('loadingSpinner').classList.remove('d-none');
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('tryNormalBtn').disabled = true;
    document.getElementById('tryPneumoniaBtn').disabled = true;
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
        document.getElementById('tryNormalBtn').disabled = false;
        document.getElementById('tryPneumoniaBtn').disabled = false;
    }
}

function showResults(data) {
    document.getElementById('resultsPanel').classList.remove('d-none');

    const predEl = document.getElementById('predClass');
    predEl.textContent = data.predicted_class;
    predEl.className = data.predicted_class === 'Pneumonia' ? 'pred-pneumonia' : 'pred-normal';

    document.getElementById('predConf').textContent = data.confidence;

    const normalProb = data.probabilities.Normal || 0;
    const pneumoniaProb = data.probabilities.Pneumonia || 0;
    document.getElementById('probNormal').style.width = normalProb + '%';
    document.getElementById('probNormal').textContent = 'Normal ' + normalProb + '%';
    document.getElementById('probPneumonia').style.width = pneumoniaProb + '%';
    document.getElementById('probPneumonia').textContent = 'Pneumonia ' + pneumoniaProb + '%';

    if (data.gradcam_image) {
        document.getElementById('gradcamImg').src = 'data:image/png;base64,' + data.gradcam_image;
    }

    document.getElementById('reportText').textContent = data.report;
}
