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

// Video processing metrics
let totalFacesDetected = 0;
let totalFacesRecognized = 0;
let recognizedFacesMap = new Map(); // name -> count
let videoStartTime = 0;

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
let currentPage = 1;
let totalPages = 1;

async function loadFaces(page = 1) {
    const grid = document.getElementById('facesGrid');
    grid.innerHTML = '<p class="empty-state">Loading faces...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/faces?page=${page}&limit=20`);
        const data = await response.json();
        
        faces = data.faces || [];
        currentPage = data.pagination?.page || 1;
        totalPages = data.pagination?.total_pages || 1;
        
        if (faces.length === 0) {
            grid.innerHTML = '<p class="empty-state">No faces enrolled yet. Go to Enroll Face to add one.</p>';
            return;
        }
        
        renderFacesList(faces);
        renderPaginationControls();
        
    } catch (error) {
        console.error('Failed to load faces:', error);
        grid.innerHTML = '<p class="empty-state">Error loading faces</p>';
    }
}

function renderFacesList(facesList) {
    const grid = document.getElementById('facesGrid');
    
    if (facesList.length === 0) {
        grid.innerHTML = '<p class="empty-state">No matching faces found</p>';
        return;
    }
    
    grid.innerHTML = `
        <table class="faces-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Image</th>
                    <th>Name</th>
                    <th>Created</th>
                    <th>Seen</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                ${facesList.map(face => `
                    <tr>
                        <td>${face.id}</td>
                        <td>
                            ${face.has_image ? 
                                `<img src="/api/faces/image/${face.id}" alt="${face.name}" class="face-thumb" onerror="this.style.display='none'">` : 
                                '<span class="no-image">👤</span>'}
                        </td>
                        <td>${escapeHtml(face.name)}</td>
                        <td>${face.created_at ? face.created_at.substring(0, 16) : '-'}</td>
                        <td>${face.seen_count || 0}</td>
                        <td>
                            <button class="btn btn-secondary btn-small" onclick="viewFace(${face.id})">View</button>
                            <button class="btn btn-danger btn-small" onclick="confirmDeleteFace(${face.id}, '${escapeHtml(face.name)}')">Delete</button>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function renderPaginationControls() {
    const grid = document.getElementById('facesGrid');
    const paginationContainer = document.createElement('div');
    paginationContainer.className = 'pagination-controls';
    
    const startFace = (currentPage - 1) * 20 + 1;
    const endFace = Math.min(currentPage * 20, startFace + faces.length - 1);
    
    paginationContainer.innerHTML = `
        <div class="pagination-info">
            Showing faces ${startFace} to ${endFace}
        </div>
        <div class="pagination-buttons">
            <button class="btn btn-secondary btn-small" onclick="goToPage(1)" ${currentPage === 1 ? 'disabled' : ''}>
                First
            </button>
            <button class="btn btn-secondary btn-small" onclick="goToPage(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}>
                Previous
            </button>
            <span class="page-indicator">Page ${currentPage} of ${totalPages}</span>
            <button class="btn btn-secondary btn-small" onclick="goToPage(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}>
                Next
            </button>
            <button class="btn btn-secondary btn-small" onclick="goToPage(${totalPages})" ${currentPage === totalPages ? 'disabled' : ''}>
                Last
            </button>
        </div>
    `;
    
    grid.appendChild(paginationContainer);
}

function goToPage(page) {
    if (page >= 1 && page <= totalPages && page !== currentPage) {
        loadFaces(page);
    }
}

function refreshFaces() {
    // Clear search input
    const searchInput = document.getElementById('searchFaces');
    if (searchInput) {
        searchInput.value = '';
    }
    loadFaces(1);
}

// Search faces
document.getElementById('searchFaces')?.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase().trim();
    const grid = document.getElementById('facesGrid');
    
    if (!query) {
        renderFacesList(faces);
        renderPaginationControls();
        return;
    }
    
    // Check if query is a number (ID search)
    const queryNum = parseInt(query);
    const isIdSearch = !isNaN(queryNum);
    
    const filtered = faces.filter(face => {
        if (isIdSearch) {
            return face.id === queryNum;
        }
        return face.name.toLowerCase().includes(query);
    });
    
    renderFacesList(filtered);
    // Hide pagination when searching within current page
    const paginationEl = grid.querySelector('.pagination-controls');
    if (paginationEl) {
        paginationEl.style.display = 'none';
    }
});

// Add search by image functionality
async function searchByImage(imageData) {
    const grid = document.getElementById('facesGrid');
    grid.innerHTML = '<p class="empty-state">Searching...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/faces/search?threshold=0.5&limit=3`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Search failed');
        }
        
        const result = await response.json();
        
        if (result.matches && result.matches.length > 0) {
            grid.innerHTML = `
                <div class="search-results">
                    <h3>Top ${result.total_matches} Matches</h3>
                    <table class="faces-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Image</th>
                                <th>Name</th>
                                <th>Similarity</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${result.matches.map(face => `
                                <tr>
                                    <td>${face.id}</td>
                                    <td>${face.has_image ? 
                                        `<img src="/api/faces/image/${face.id}" alt="${face.name}" class="face-thumb">` : 
                                        '<span class="no-image">👤</span>'}</td>
                                    <td>${escapeHtml(face.name)}</td>
                                    <td><span class="similarity-badge">${(face.similarity * 100).toFixed(1)}%</span></td>
                                    <td>
                                        <button class="btn btn-secondary btn-small" onclick="viewFace(${face.id})">View</button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        } else {
            grid.innerHTML = '<p class="empty-state">No matching faces found</p>';
        }
        
    } catch (error) {
        console.error('Search failed:', error);
        grid.innerHTML = `<p class="empty-state">Search failed: ${escapeHtml(error.message)}</p>`;
    }
}

function viewFace(faceId) {
    // Show loading first
    showModal('Face Details', '<p class="empty-state">Loading...</p>');
    
    // Fetch face details from API
    fetch(`${API_BASE}/faces/${faceId}`)
        .then(response => {
            if (!response.ok) throw new Error('Face not found');
            return response.json();
        })
        .then(face => {
            const modalContent = `
                <div class="face-view-modal">
                    <div class="face-view-image">
                        ${face.has_image ? 
                            `<img src="/api/faces/image/${face.id}" alt="${face.name}">` : 
                            '<div class="no-image-large">👤</div>'}
                    </div>
                    <div class="face-view-details">
                        <h2>${escapeHtml(face.name)}</h2>
                        <table class="detail-table">
                            <tr><td><strong>ID:</strong></td><td>${face.id}</td></tr>
                            <tr><td><strong>Created:</strong></td><td>${face.created_at || '-'}</td></tr>
                            <tr><td><strong>Last Seen:</strong></td><td>${face.last_seen_at || 'Never'}</td></tr>
                            <tr><td><strong>Times Seen:</strong></td><td>${face.seen_count || 0}</td></tr>
                        </table>
                        <button class="btn btn-danger" onclick="confirmDeleteFace(${face.id}, '${escapeHtml(face.name)}'); closeModal();">Delete Face</button>
                    </div>
                </div>
            `;
            showModal('Face Details', modalContent);
        })
        .catch(error => {
            showModal('Error', `<p class="empty-state">${escapeHtml(error.message)}</p>`);
        });
}

// Enroll Form
function initForms() {
    const enrollForm = document.getElementById('enrollForm');
    if (enrollForm) {
        enrollForm.addEventListener('submit', handleEnrollSubmit);
    }
    
    // Image upload preview (no drag-and-drop, just file selection)
    const imageUpload = document.getElementById('faceImage');
    const uploadArea = document.getElementById('imageUploadArea');
    const previewImage = document.getElementById('previewImage');
    
    if (imageUpload && uploadArea) {
        // Click to select file
        imageUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleImageFile(file, previewImage, uploadArea);
            }
        });
        
        // Click on upload area to trigger file input
        uploadArea.addEventListener('click', (e) => {
            if (e.target !== imageUpload) {
                imageUpload.click();
            }
        });
    }
    
    // Helper function to handle image file
    function handleImageFile(file, previewImage, uploadArea) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPG, PNG, BMP)');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadArea.classList.add('has-image');
        };
        reader.onerror = () => {
            alert('Error reading image file');
        };
        reader.readAsDataURL(file);
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
            // Check if browser supports mediaDevices
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                statusEl.textContent = 'Camera access not supported by browser';
                alert('Your browser does not support camera access. Try Chrome or Firefox with HTTPS.');
                return;
            }
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        frameRate: { ideal: 15, max: 30 } // Limit frame rate
                    } 
                });
                videoElement.srcObject = stream;
                videoStream = stream;
                
                // Wait for video to be ready
                await new Promise((resolve) => {
                    videoElement.onloadedmetadata = resolve;
                });
            } catch (cameraError) {
                console.error('Camera access failed:', cameraError);
                
                // Provide user-friendly error messages
                let errorMessage = 'Camera error: ';
                if (cameraError.name === 'NotAllowedError') {
                    errorMessage += 'Camera access denied. Please allow camera permissions.';
                } else if (cameraError.name === 'NotFoundError') {
                    errorMessage += 'No camera found. Please connect a camera.';
                } else if (cameraError.name === 'NotReadableError') {
                    errorMessage += 'Camera is already in use by another application.';
                } else {
                    errorMessage += cameraError.message;
                }
                
                statusEl.textContent = errorMessage;
                alert(errorMessage);
                return;
            }
        } else {
            const videoFile = document.getElementById('videoFile').files[0];
            if (!videoFile) {
                alert('Please select a video file');
                return;
            }
            
            // Create object URL and set source
            const objectUrl = URL.createObjectURL(videoFile);
            videoElement.src = objectUrl;
            
            // Clean up object URL when video ends or on error
            videoElement.onended = () => URL.revokeObjectURL(objectUrl);
            videoElement.onerror = () => URL.revokeObjectURL(objectUrl);
            
            // Try to play with error handling
            try {
                await videoElement.play();
            } catch (playError) {
                console.warn('Video play failed, trying again:', playError);
                
                // Common error: "play() request was interrupted"
                // Wait a bit and try again
                await new Promise(resolve => setTimeout(resolve, 100));
                
                try {
                    await videoElement.play();
                } catch (retryError) {
                    console.error('Video play retry failed:', retryError);
                    statusEl.textContent = 'Failed to play video: ' + retryError.message;
                    
                    // Clean up and return
                    URL.revokeObjectURL(objectUrl);
                    return;
                }
            }
        }
        
        // Reset metrics
        totalFacesDetected = 0;
        totalFacesRecognized = 0;
        recognizedFacesMap.clear();
        videoStartTime = Date.now();
        
        // Clear recognized faces list
        const recognizedList = document.getElementById('recognizedFacesList');
        if (recognizedList) {
            recognizedList.innerHTML = '<li class="text-gray-500">No faces recognized yet</li>';
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
    
    // Stop camera stream if active
    if (videoStream) {
        videoStream.getTracks().forEach(track => {
            track.stop();
            track.enabled = false;
        });
        videoStream = null;
    }
    
    // Pause video and clean up
    videoElement.pause();
    videoElement.currentTime = 0;
    
    // Clean up object URL if used
    if (videoElement.src && videoElement.src.startsWith('blob:')) {
        URL.revokeObjectURL(videoElement.src);
    }
    
    videoElement.srcObject = null;
    videoElement.src = '';
    videoProcessing = false;
    
    // Reset processing state and metrics
    isProcessingFrame = false;
    frameQueue = [];
    lastFrameTime = 0;
    minFrameInterval = MIN_FRAME_INTERVAL_MIN;
    totalFacesDetected = 0;
    totalFacesRecognized = 0;
    recognizedFacesMap.clear();
    videoStartTime = 0;
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusEl.textContent = 'Stopped';
    
    // Clear canvas
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Reset stats
    document.getElementById('videoFps').textContent = '0';
    document.getElementById('facesDetected').textContent = '0';
    document.getElementById('facesRecognized').textContent = '0';
}

// Track processing state to prevent overlapping requests
let isProcessingFrame = false;
let frameQueue = [];
let lastFrameTime = 0;
let minFrameInterval = 200; // Process at most 5 FPS (1000ms / 5 = 200ms)
const MIN_FRAME_INTERVAL_MIN = 200;
const MIN_FRAME_INTERVAL_MAX = 1000;

async function processVideoFrame() {
    if (!videoProcessing) return;
    
    const now = Date.now();
    const timeSinceLastFrame = now - lastFrameTime;
    
    // Skip if we're processing too fast or already processing a frame
    if (isProcessingFrame || timeSinceLastFrame < minFrameInterval) {
        // Schedule next check
        setTimeout(processVideoFrame, 50);
        return;
    }
    
    isProcessingFrame = true;
    lastFrameTime = now;
    
    const videoElement = document.getElementById('videoElement');
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    const statusEl = document.getElementById('videoStatus');
    
    // Clear canvas before drawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA && 
        videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
        
        // Set canvas size to match video
        if (canvas.width !== videoElement.videoWidth || canvas.height !== videoElement.videoHeight) {
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
        }
        
        // Draw current video frame
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        const enableRecognition = document.getElementById('enableRecognition').checked;
        
        try {
            // Capture frame from canvas as base64 with lower quality for faster transfer
            const imageData = canvas.toDataURL('image/jpeg', 0.5);
            
            // Add timeout to prevent hanging requests
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch(`${API_BASE}/video/detect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    frame: imageData,
                    recognize: enableRecognition
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            const data = await response.json();
            
            // Clear canvas and redraw video frame with overlays
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            
            if (data.faces && data.faces.length > 0) {
                // Draw face boxes
                data.faces.forEach(face => {
                    const box = face.bounding_box;
                    ctx.strokeStyle = face.name ? '#10b981' : '#ef4444';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(box.x, box.y, box.width, box.height);
                    
                    // Draw label
                    ctx.fillStyle = face.name ? '#10b981' : '#ef4444';
                    ctx.fillRect(box.x, box.y - 25, Math.max(box.width, 80), 25);
                    ctx.fillStyle = 'white';
                    ctx.font = '14px sans-serif';
                    const label = face.name || 'Unknown';
                    ctx.fillText(label, box.x + 5, box.y - 7);
                    
                    // Update accumulated metrics
                    totalFacesDetected++;
                    if (face.name) {
                        totalFacesRecognized++;
                        // Track recognized faces by name
                        recognizedFacesMap.set(face.name, (recognizedFacesMap.get(face.name) || 0) + 1);
                    }
                });
                
                // Update UI with accumulated totals
                document.getElementById('facesDetected').textContent = totalFacesDetected;
                document.getElementById('facesRecognized').textContent = totalFacesRecognized;
                
                // Update recognized faces list
                updateRecognizedFacesList();
            }
            
            // Show current FPS (not accumulated)
            document.getElementById('videoFps').textContent = data.fps || '0';
            
            // Calculate and show elapsed time
            const elapsedSeconds = Math.floor((Date.now() - videoStartTime) / 1000);
            const minutes = Math.floor(elapsedSeconds / 60);
            const seconds = elapsedSeconds % 60;
            statusEl.textContent = `Processing ${data.fps || '0'} FPS | Time: ${minutes}:${seconds.toString().padStart(2, '0')}`;
            
        } catch (error) {
            if (error.name === 'AbortError') {
                console.warn('Frame processing timeout');
                statusEl.textContent = 'Processing timeout - slowing down';
                // Increase interval on timeout (with bounds)
                minFrameInterval = Math.min(minFrameInterval * 1.5, MIN_FRAME_INTERVAL_MAX);
            } else {
                console.error('Frame processing error:', error);
                statusEl.textContent = 'Error processing frame';
            }
            
            // Clear stats on error
            document.getElementById('facesDetected').textContent = '0';
            document.getElementById('facesRecognized').textContent = '0';
        }
    } else {
        statusEl.textContent = 'Waiting for video data...';
    }
    
    isProcessingFrame = false;
    
    // Continue loop with adaptive timing
    const nextDelay = Math.max(50, minFrameInterval - (Date.now() - now));
    setTimeout(processVideoFrame, nextDelay);
}

function updateRecognizedFacesList() {
    const recognizedList = document.getElementById('recognizedFacesList');
    if (!recognizedList) return;
    
    if (recognizedFacesMap.size === 0) {
        recognizedList.innerHTML = '<li class="text-gray-500">No faces recognized yet</li>';
        return;
    }
    
    // Sort by count (descending)
    const sortedFaces = Array.from(recognizedFacesMap.entries())
        .sort((a, b) => b[1] - a[1]);
    
    // Update list
    recognizedList.innerHTML = sortedFaces.map(([name, count]) => 
        `<li class="flex justify-between items-center py-1">
            <span class="font-medium">${name}</span>
            <span class="bg-green-100 text-green-800 text-xs font-medium px-2 py-0.5 rounded">${count}×</span>
         </li>`
    ).join('');
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

// Modal functions
function showModal(title, content) {
    // Remove existing modal if any
    const existing = document.getElementById('customModal');
    if (existing) existing.remove();
    
    const modal = document.createElement('div');
    modal.id = 'customModal';
    modal.className = 'modal active';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="close-modal" onclick="closeModal()">&times;</span>
            <h2>${title}</h2>
            ${content}
        </div>
    `;
    document.body.appendChild(modal);
    
    // Close on outside click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
}

function closeModal() {
    const modal = document.getElementById('customModal');
    if (modal) modal.remove();
}

// Handle image search upload
function handleSearchImageUpload(input) {
    if (!input.files || !input.files[0]) return;
    
    const file = input.files[0];
    const reader = new FileReader();
    
    reader.onload = function(e) {
        // Convert to base64
        const imageData = e.target.result.split(',')[1]; // Remove data URL prefix
        searchByImage(imageData);
    };
    
    reader.readAsDataURL(file);
    // Reset input so same file can be selected again
    input.value = '';
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
