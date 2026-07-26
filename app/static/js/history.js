// history page js

document.addEventListener('DOMContentLoaded', loadHistory);

async function loadHistory() {
    try {
        const res = await fetch('/api/v1/history?limit=50');
        const data = await res.json();

        document.getElementById('historyLoading').classList.add('d-none');

        if (data.length === 0) {
            document.getElementById('historyEmpty').classList.remove('d-none');
            return;
        }

        const tbody = document.getElementById('historyBody');
        data.forEach(item => {
            const tr = document.createElement('tr');
            const date = new Date(item.created_at).toLocaleString();
            const badgeClass = item.predicted_class === 'Pneumonia' ? 'bg-danger' : 'bg-success';

            tr.innerHTML = `
                <td>${item.id}</td>
                <td>${item.filename}</td>
                <td><span class="badge ${badgeClass}">${item.predicted_class}</span></td>
                <td>${item.confidence}%</td>
                <td>${date}</td>
                <td><button class="btn btn-sm btn-outline-primary" onclick="viewReport(${item.id})">View Report</button></td>
            `;
            tbody.appendChild(tr);
        });

        document.getElementById('historyTable').classList.remove('d-none');

    } catch (err) {
        document.getElementById('historyLoading').classList.add('d-none');
        alert('Failed to load history');
    }
}

async function viewReport(id) {
    try {
        const res = await fetch('/api/v1/history/' + id);
        const data = await res.json();
        document.getElementById('modalReport').textContent = data.report || 'No report saved';
        new bootstrap.Modal(document.getElementById('reportModal')).show();
    } catch (err) {
        alert('Could not load report');
    }
}
