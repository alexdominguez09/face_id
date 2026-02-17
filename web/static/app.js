// Face Recognition System - Frontend JavaScript

// API Base URL
const API_BASE = '/api';

// Global state
let currentView = 'dashboard';
let faces = [];
let videoStream = null;
let videoProcessing = false;
let detectionInterval = null;
let faceToDelete = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initForms();
    checkSystemStatus();
    loadDashboard();
});

// Navigation
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            switchView(view);
        });
    });
}

function switchView(view) {
    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === view);
    });
    
    // Update views
    document.querySelectorAll('.view').forEach(v => {
        v.classList.toggle('active', v.id === `${view}-view`);
    });
    
    currentView = view;
    
    // Load view data
    switch (view) {
        case 'dashboard':
            loadDashboard();
            break;
        case 'faces':
            loadFaces();
            break;
        case 'enroll':
            clearEnrollForm();
            break;
        case 'video':
            // Video view initialized
            break;
        case 'settings':
            loadSettings();
            break;
    }
}

// System Status
async function checkSystemStatus() {
    const statusEl = document.getElementById('systemStatus');
    
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            statusEl.className = 'status-indicator status-online';
            statusEl.textContent = 'Online';
        } else {
            throw new Error('Health check failed');
        }
    } catch (error) {
        statusEl.className = 'status-indicator status-offline';
        statusEl.textContent = 'Offline';
    }
}

// Dashboard
async function loadDashboard() {
    try {
        const [statsRes, historyRes] = await Promise.all([
            fetch(`${API_BASE}/stats`),
            fetch(`${API_BASE}/activity`)
        ]);
        
        const stats = await statsRes.json();
        const history = await historyRes.json();
        
        // Update stats
        document.getElementById('totalFaces').textContent = stats.total_faces || 0;
        document.getElementById('detectionsToday').textContent = stats.detections_today || 0;
        document.getElementById('avgProcessingTime').textContent = stats.avg_processing_ms ? `${Math.round(stats.avg_processing_ms)}ms` : '-';
        document.getElementById('systemUptime').textContent = stats.uptime || '-';
        
        // Update activity
        const activityList = document.getElementById('activityList');
        if (history.activities && history.activities.length > 0) {
            activityList.innerHTML = history.activities.slice(0, 10).map(activity => `
                <div class="activity-item">
                    <div class="activity-icon ${activity.type}">${getActivityIcon(activity.type)}</div>
                    <div class="activity-content">
                        <div class="activity-text">${activity.message}</div>
                        <div class="activity-time">${formatTime(activity.timestamp)}</div>
                    </div>
                </div>
            `).join('');
        } else {
            activityList.innerHTML = '<p class="empty-state">No recent activity</p>';
        }
    } catch (error) {
        console.error('Failed to load dashboard:', error);
    }
}

function getActivityIcon(type) {
    const icons = {
        'detection': '👤',
        'recognition': '🎯',
        'enrollment': '➕',
        'deletion': '🗑️',
        'error': '⚠️',
        'info': 'ℹ️'
    };
    return icons[type] || 'ℹ️';
}

function formatTime(timestamp) {
    if (!timestamp) return 'Just now';
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
}

// Faces
async function loadFaces() {
    const grid = document.getElementById('facesGrid');
    grid.innerHTML = '<p class="empty-state">Loading faces...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/faces`);
        const data = await response.json();
        
        faces = data.faces || [];
        
        if (faces.length === 0) {
            grid.innerHTML = '<p class="empty-state">No faces enrolled yet. Go to Enroll Face to add one.</p>';
            return;
        }
        
        grid.innerHTML = faces.map(face => `
            <div class="face-card" data-face-id="${face.id}">
                <div class="face-image">
                    ${face.image_path ? `<img src="/api/faces/image/${face.id}" alt="${face.name}" onerror="this.parentElement.innerHTML='👤'">` : '👤'}
                </div>
                <div class="face-info">
                    <div class="face-name">${escapeHtml(face.name)}</div>
                    <div class="face-meta">
                        ID: ${face.id} • Seen ${face.seen_count || 0} times
                    </div>
                </div>
                <div class="face-actions">
                    <button class="btn btn-secondary btn-small" onclick="viewFace(${face.id})">View</button>
                    <button class="btn btn-danger btn-small" onclick="confirmDeleteFace(${face.id}, '${escapeHtml(face.name)}')">Delete</button>
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Failed to load faces:', error);
        grid.innerHTML = '<p class="empty-state">Error loading faces</p>';
    }
}

function refreshFaces() {
    loadFaces();
}

// Search faces
document.getElementById('searchFaces')?.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    const grid = document.getElementById('facesGrid');
    
    const filtered = faces.filter(face => 
        face.name.toLowerCase().includes(query)
    );
    
    if (filtered.length === 0) {
        grid.innerHTML = '<p class="empty-state">No matching faces found</p>';
        return;
    }
    
    grid.innerHTML = filtered.map(face => `
        <div class="face-card" data-face-id="${face.id}">
            <div class="face-image">
                ${face.image_path ? `<img src="/api/faces/image/${face.id}" alt="${face.name}" onerror="this.parentElement.innerHTML='👤'">` : '👤'}
            </div>
            <div class="face-info">
                <div class="face-name">${escapeHtml(face.name)}</div>
                <div class="face-meta">
                    ID: ${face.id} • Seen ${face.seen_count || 0} times
                </div>
            </div>
            <div class="face-actions">
                <button class="btn btn-secondary btn-small" onclick="viewFace(${face.id})">View</button>
                <button class="btn btn-danger btn-small" onclick="confirmDeleteFace(${face.id}, '${escapeHtml(face.name)}')">Delete</button>
            </div>
        </div>
    `).join('');
});

function viewFace(faceId) {
    window.location.href = `/api/faces/${faceId}`;
}

// Enroll Form
function initForms() {
    const enrollForm = document.getElementById('enrollForm');
    if (enrollForm) {
        enrollForm.addEventListener('submit', handleEnrollSubmit);
    }
    
    // Image upload preview
    const imageUpload = document.getElementById('faceImage');
    const uploadArea = document.getElementById('imageUploadArea');
    const previewImage = document.getElementById('previewImage');
    
    if (imageUpload && uploadArea) {
        imageUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    uploadArea.classList.add('has-image');
                };
                reader.readAsDataURL(file);
            }
        });
    }
    
    // Video source toggle
    const videoSource = document.getElementById('videoSource');
    const videoFileGroup = document.getElementById('videoFileGroup');
    if (videoSource && videoFileGroup) {
        videoSource.addEventListener('change', (e) => {
            videoFileGroup.style.display = e.target.value === 'file' ? 'block' : 'none';
        });
    }
    
    // Settings sliders
    const recognitionThreshold = document.getElementById('recognitionThreshold');
    if (recognitionThreshold) {
        recognitionThreshold.addEventListener('input', (e) => {
            document.getElementById('recognitionThresholdValue').textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    const duplicateThreshold = document.getElementById('duplicateThreshold');
    if (duplicateThreshold) {
        duplicateThreshold.addEventListener('input', (e) => {
            document.getElementById('duplicateThresholdValue').textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
}

async function handleEnrollSubmit(e) {
    e.preventDefault();
    
    const form = e.target;
    const resultEl = document.getElementById('enrollResult');
    const submitBtn = document.getElementById('enrollBtn');
    
    const name = document.getElementById('personName').value.trim();
    const imageFile = document.getElementById('faceImage').files[0];
    const metadata = document.getElementById('metadata').value.trim();
    
    if (!name || !imageFile) {
        showResult(resultEl, 'Please provide a name and image', 'error');
        return;
    }
    
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="loading"></span> Enrolling...';
    
    try {
        // Convert image to base64
        const base64Image = await fileToBase64(imageFile);
        
        const response = await fetch(`${API_BASE}/faces/enroll`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                image: base64Image,
                metadata: metadata || undefined
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showResult(resultEl, `Successfully enrolled ${name}! (Face ID: ${data.face_id})`, 'success');
            clearEnrollForm();
            addActivity('enrollment', `Enrolled new face: ${name}`);
        } else {
            showResult(resultEl, data.detail || 'Failed to enroll face', 'error');
        }
    } catch (error) {
        console.error('Enrollment error:', error);
        showResult(resultEl, 'Error enrolling face: ' + error.message, 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Enroll Face';
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function clearEnrollForm() {
    const form = document.getElementById('enrollForm');
    if (form) {
        form.reset();
    }
    
    const uploadArea = document.getElementById('imageUploadArea');
    const previewImage = document.getElementById('previewImage');
    if (uploadArea && previewImage) {
        uploadArea.classList.remove('has-image');
        previewImage.src = '';
    }
    
    const resultEl = document.getElementById('enrollResult');
    if (resultEl) {
        resultEl.className = 'result-message';
        resultEl.textContent = '';
    }
}

function showResult(element, message, type) {
    element.className = `result-message ${type}`;
    element.textContent = message;
}

// Delete Face
function confirmDeleteFace(faceId, faceName) {
    faceToDelete = faceId;
    const modal = document.getElementById('deleteModal');
    const message = document.getElementById('deleteMessage');
    
    message.textContent = `Are you sure you want to delete "${faceName}"? This action cannot be undone.`;
    modal.classList.add('active');
    
    document.getElementById('confirmDeleteBtn').onclick = () => deleteFace(faceId);
}

function closeDeleteModal() {
    const modal = document.getElementById('deleteModal');
    modal.classList.remove('active');
    faceToDelete = null;
}

async function deleteFace(faceId) {
    try {
        const response = await fetch(`${API_BASE}/faces/${faceId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            closeDeleteModal();
            loadFaces();
            addActivity('deletion', `Deleted face ID: ${faceId}`);
        } else {
            const data = await response.json();
            alert(data.detail || 'Failed to delete face');
        }
    } catch (error) {
        console.error('Delete error:', error);
        alert('Error deleting face');
    }
}

// Video Processing
async function startVideo() {
    const videoElement = document.getElementById('videoElement');
    const startBtn = document.getElementById('startVideoBtn');
    const stopBtn = document.getElementById('stopVideoBtn');
    const statusEl = document.getElementById('videoStatus');
    
    const source = document.getElementById('videoSource').value;
    const enableRecognition = document.getElementById('enableRecognition').checked;
    
    try {
        if (source === 'camera') {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            videoElement.srcObject = stream;
            videoStream = stream;
        } else {
            const videoFile = document.getElementById('videoFile').files[0];
            if (!videoFile) {
                alert('Please select a video file');
                return;
            }
            videoElement.src = URL.createObjectURL(videoFile);
            await videoElement.play();
        }
        
        videoProcessing = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusEl.textContent = 'Processing...';
        
        // Start processing loop
        processVideoFrame();
        
    } catch (error) {
        console.error('Video start error:', error);
        statusEl.textContent = 'Error: ' + error.message;
    }
}

function stopVideo() {
    const videoElement = document.getElementById('videoElement');
    const startBtn = document.getElementById('startVideoBtn');
    const stopBtn = document.getElementById('stopVideoBtn');
    const statusEl = document.getElementById('videoStatus');
    
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    
    videoElement.srcObject = null;
    videoProcessing = false;
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusEl.textContent = 'Stopped';
    
    // Reset stats
    document.getElementById('videoFps').textContent = '0';
    document.getElementById('facesDetected').textContent = '0';
    document.getElementById('facesRecognized').textContent = '0';
}

async function processVideoFrame() {
    if (!videoProcessing) return;
    
    const videoElement = document.getElementById('videoElement');
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    const statusEl = document.getElementById('videoStatus');
    
    if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        ctx.drawImage(videoElement, 0, 0);
        
        const enableRecognition = document.getElementById('enableRecognition').checked;
        
        try {
            const response = await fetch(`${API_BASE}/video/detect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    frame: videoElement.src,
                    recognize: enableRecognition
                })
            });
            
            const data = await response.json();
            
            if (data.faces && data.faces.length > 0) {
                // Draw face boxes
                data.faces.forEach(face => {
                    const box = face.bounding_box;
                    ctx.strokeStyle = face.name ? '#10b981' : '#ef4444';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(box.x, box.y, box.width, box.height);
                    
                    // Draw label
                    ctx.fillStyle = face.name ? '#10b981' : '#ef4444';
                    ctx.fillRect(box.x, box.y - 25, box.width, 25);
                    ctx.fillStyle = 'white';
                    ctx.font = '14px sans-serif';
                    ctx.fillText(face.name || 'Unknown', box.x + 5, box.y - 7);
                });
                
                document.getElementById('facesDetected').textContent = data.faces.length;
                document.getElementById('facesRecognized').textContent = data.faces.filter(f => f.name).length;
            } else {
                document.getElementById('facesDetected').textContent = '0';
            }
            
            document.getElementById('videoFps').textContent = data.fps || '0';
            statusEl.textContent = 'Processing...';
            
        } catch (error) {
            console.error('Frame processing error:', error);
        }
    }
    
    // Continue loop
    setTimeout(processVideoFrame, 100);
}

// Settings
async function loadSettings() {
    try {
        const response = await fetch(`${API_BASE}/settings`);
        const settings = await response.json();
        
        document.getElementById('dbFaceCount').textContent = settings.total_faces || '0';
        document.getElementById('recognitionThreshold').value = settings.recognition_threshold || 0.5;
        document.getElementById('recognitionThresholdValue').textContent = (settings.recognition_threshold || 0.5).toFixed(2);
        document.getElementById('duplicateThreshold').value = settings.duplicate_threshold || 0.85;
        document.getElementById('duplicateThresholdValue').textContent = (settings.duplicate_threshold || 0.85).toFixed(2);
    } catch (error) {
        console.error('Failed to load settings:', error);
    }
}

async function exportDatabase() {
    try {
        const response = await fetch(`${API_BASE}/export`);
        const data = await response.json();
        
        // Download as JSON
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'face_database_export.json';
        a.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        alert('Failed to export database');
    }
}

async function clearDatabase() {
    if (!confirm('Are you sure you want to delete ALL faces? This cannot be undone!')) {
        return;
    }
    
    if (!confirm('This will remove all enrolled faces. Continue?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/faces/clear`, {
            method: 'POST'
        });
        
        if (response.ok) {
            alert('All faces cleared successfully');
            loadSettings();
            addActivity('deletion', 'Cleared all faces from database');
        } else {
            alert('Failed to clear database');
        }
    } catch (error) {
        alert('Error clearing database');
    }
}

// Activity tracking (local)
function addActivity(type, message) {
    const activity = {
        type: type,
        message: message,
        timestamp: new Date().toISOString()
    };
    
    // Store in localStorage for persistence
    let activities = JSON.parse(localStorage.getItem('face_activities') || '[]');
    activities.unshift(activity);
    activities = activities.slice(0, 100); // Keep last 100
    localStorage.setItem('face_activities', JSON.stringify(activities));
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Close modal on outside click
document.getElementById('deleteModal')?.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        closeDeleteModal();
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeDeleteModal();
    }
});
