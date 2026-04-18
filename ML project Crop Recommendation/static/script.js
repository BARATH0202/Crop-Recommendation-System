// Global State
let currentPrediction = null;
let currentSoilData = [];
let npkChartInstance = null;
let envChartInstance = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Initial fetch of DB
    fetchSoilData();
});

// UI Navigation
function showSection(sectionId) {
    document.querySelectorAll('.content-section').forEach(sec => sec.classList.remove('active'));
    document.getElementById(sectionId).classList.add('active');
    
    // Refresh components if needed
    if(sectionId === 'database-section') {
        fetchSoilData();
    } else if(sectionId === 'analytics-section') {
        fetchAnalytics();
    }
}

// Toast Notification
function showToast(message, isError = false) {
    const toastEl = document.getElementById('actionToast');
    const toastBody = document.getElementById('toastMessage');
    toastBody.textContent = message;
    
    if(isError) {
        toastEl.classList.remove('bg-success');
        toastEl.classList.add('bg-danger');
    } else {
        toastEl.classList.add('bg-success');
        toastEl.classList.remove('bg-danger');
    }
    
    const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
    toast.show();
}

// Prediction Form
document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = document.getElementById('predictBtn');
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
    btn.disabled = true;

    // Collect data
    const formData = new FormData();
    const imageFile = document.getElementById('soilImage').files[0];
    
    if(!imageFile) {
        showToast('Please upload a soil image', true);
        btn.innerHTML = '<i class="fas fa-magic me-2"></i>Analyze & Predict';
        btn.disabled = false;
        return;
    }
    
    formData.append('image', imageFile);
    formData.append('temperature', document.getElementById('temperature').value);
    formData.append('humidity', document.getElementById('humidity').value);
    formData.append('rainfall', document.getElementById('rainfall').value);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData // No headers required for FormData
        });
        const data = await response.json();

        if(!response.ok) throw new Error(data.error || 'Server error');

        // Show Result
        currentPrediction = {
            temperature: document.getElementById('temperature').value,
            humidity: document.getElementById('humidity').value,
            rainfall: document.getElementById('rainfall').value,
            soil_type: data.soil_type,
            recommended_crop: data.recommended_crop
        };
        
        document.getElementById('soilAnalysisBadge').textContent = `${data.soil_type.toUpperCase()} (${data.soil_confidence})`;
        document.getElementById('cropName').textContent = data.recommended_crop;
        document.getElementById('confidenceBadge').textContent = data.crop_confidence + ' Crop Confidence';
        
        const imgEl = document.getElementById('cropImage');
        // Produce a dynamic, open-source image representation of the crop using Pollinations.ai 
        imgEl.src = `https://pollinations.ai/p/${encodeURIComponent(data.recommended_crop + " fruit or crop agriculture field realistic photography")}?width=300&height=200&nologo=true`;
        imgEl.classList.remove('d-none');
        
        document.getElementById('resultCard').classList.remove('d-none');
        
        showToast('Analysis completed successfully!');
    } catch (error) {
        showToast(error.message, true);
    } finally {
        btn.innerHTML = '<i class="fas fa-magic me-2"></i>Analyze & Predict';
        btn.disabled = false;
    }
});

// Save Prediction to DB
async function saveToDatabase() {
    if(!currentPrediction) return;

    try {
        const response = await fetch('/soil', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentPrediction)
        });
        const data = await response.json();

        if(!response.ok) throw new Error(data.error);

        showToast('Record saved to Database!');
        fetchSoilData(); // update state in background
    } catch (error) {
        showToast('Failed to save record.', true);
    }
}

// Fetch All Soil Data
async function fetchSoilData() {
    try {
        const response = await fetch('/soil');
        const data = await response.json();
        
        if(!response.ok) throw new Error('Failed to fetch data');
        currentSoilData = data;
        
        document.getElementById('db-count').textContent = data.length;
        renderTable(data);
    } catch (error) {
        console.error(error);
    }
}

// Render Table
function renderTable(data) {
    const tbody = document.getElementById('soilTableBody');
    tbody.innerHTML = '';
    
    if(data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No records found.</td></tr>';
        return;
    }

    data.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>#${row.id}</td>
            <td>${parseFloat(row.temperature).toFixed(1)}</td>
            <td>${parseFloat(row.humidity).toFixed(1)}</td>
            <td>${parseFloat(row.rainfall).toFixed(1)}</td>
            <td><span class="badge bg-secondary">${row.soil_type.toUpperCase()}</span></td>
            <td><span class="badge bg-success">${row.recommended_crop}</span></td>
            <td class="text-center">
                <button class="btn btn-sm btn-outline-info me-1" onclick="openEdit(${row.id})"><i class="fas fa-edit"></i></button>
                <button class="btn btn-sm btn-outline-danger" onclick="deleteRecord(${row.id})"><i class="fas fa-trash"></i></button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// Delete Record
async function deleteRecord(id) {
    if(!confirm('Are you sure you want to delete this record?')) return;
    
    try {
        const res = await fetch(`/soil/${id}`, { method: 'DELETE' });
        if(!res.ok) throw new Error('Failed to delete');
        
        showToast('Record deleted');
        fetchSoilData();
    } catch (error) {
        showToast(error.message, true);
    }
}

// Edit Record
let editModalInstance = null;
function openEdit(id) {
    const record = currentSoilData.find(r => r.id === id);
    if(!record) return;

    document.getElementById('edit-id').value = id;
    document.getElementById('edit-temp').value = record.temperature;
    document.getElementById('edit-humidity').value = record.humidity;
    document.getElementById('edit-rainfall').value = record.rainfall;
    document.getElementById('edit-soil').value = record.soil_type;
    document.getElementById('edit-crop').value = record.recommended_crop;

    if(!editModalInstance) {
        editModalInstance = new bootstrap.Modal(document.getElementById('editModal'));
    }
    editModalInstance.show();
}

async function submitEdit() {
    const id = document.getElementById('edit-id').value;
    const payload = {
        temperature: document.getElementById('edit-temp').value,
        humidity: document.getElementById('edit-humidity').value,
        rainfall: document.getElementById('edit-rainfall').value,
        soil_type: document.getElementById('edit-soil').value,
        recommended_crop: document.getElementById('edit-crop').value
    };

    try {
        const res = await fetch(`/soil/${id}`, {
            method: 'PUT',
            headers:{ 'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        
        if(!res.ok) throw new Error('Failed to update record');
        
        showToast('Record updated successfully!');
        editModalInstance.hide();
        fetchSoilData();
    } catch (error) {
        showToast(error.message, true);
    }
}

// Charts System
async function fetchAnalytics() {
    try {
        const response = await fetch('/analytics');
        const data = await response.json();
        if(!response.ok) throw new Error(data.error || 'Failed to fetch analytics');
        
        updateAnalyticsCharts(data);
    } catch (error) {
        showToast(error.message, true);
    }
}

function updateAnalyticsCharts(data) {
    // Destroy old if exist
    if(npkChartInstance) npkChartInstance.destroy();
    if(envChartInstance) envChartInstance.destroy();

    Chart.defaults.color = '#e0e0e0';

    // Env Radar Map
    const ctx1 = document.getElementById('npkChart').getContext('2d');
    npkChartInstance = new Chart(ctx1, {
        type: 'radar',
        data: {
            labels: ['Temperature', 'Humidity', 'Rainfall'],
            datasets: [{
                label: 'Average Environmental Stats',
                data: [data.averages.avg_T || 0, data.averages.avg_H || 0, data.averages.avg_R || 0],
                backgroundColor: 'rgba(46, 204, 161, 0.4)',
                borderColor: '#2ecca1',
                pointBackgroundColor: '#fff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            scales: {
                r: { angleLines: { color: 'rgba(255,255,255,0.1)' }, grid: { color: 'rgba(255,255,255,0.1)' } }
            },
            plugins: { title: { display: true, text: 'Overall Environmental Trends' } }
        }
    });

    // Recommendations Doughnut Chart
    const ctx2 = document.getElementById('envChart').getContext('2d');
    const cropLabels = Object.keys(data.crop_distribution);
    const cropCounts = Object.values(data.crop_distribution);
    
    envChartInstance = new Chart(ctx2, {
        type: 'doughnut',
        data: {
            labels: cropLabels,
            datasets: [{
                label: 'Crop Recommendations',
                data: cropCounts,
                backgroundColor: ['#f39c12', '#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#00c6ff', '#f1c40f'],
                borderWidth: 1,
                borderColor: '#1e293b'
            }]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: 'Crop Recommendation Distribution' } }
        }
    });
}
