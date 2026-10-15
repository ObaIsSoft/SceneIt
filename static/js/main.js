// --- Config ---
let VIDEO_TITLES = []; // Fetched from backend

// Fetch config on load
async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        VIDEO_TITLES = data.video_objects;
        console.log("Loaded video config:", VIDEO_TITLES);
    } catch (err) {
        console.error("Failed to load config:", err);
        // Fallback
        VIDEO_TITLES = [{ title: "BigBuckBunny", filename: "BigBuckBunny_CDN.mp4" }]; // Simplified fallback
    }
}
loadConfig();
let currentQueryHash = null; // Used for feedback
let selectedFile = null;

// --- Elements ---
const els = {
    form: document.getElementById('upload-form'),
    input: document.getElementById('upload-input'),
    dropZone: document.getElementById('drop-zone'),
    dropOverlay: document.getElementById('drop-overlay'),
    prompt: document.getElementById('upload-prompt'),
    previewContainer: document.getElementById('preview-container'),
    previewImg: document.getElementById('preview-img'),
    previewVideo: document.getElementById('preview-video'),
    clearBtn: document.getElementById('clear-file'),
    searchBtn: document.getElementById('search-button'),
    resultsContainer: document.getElementById('results-container'),
    resultsGrid: document.getElementById('results-grid'),
    statusMsg: document.getElementById('status-msg'),
    // NOTE: 'browse-toggle-area' and 'browse-btn' used to be looked up here and
    // do not exist in index.html. Both resolved to null, so the browse listener
    // was never attached and the header's Explore Library button -- which
    // proxied through els.browseBtn.click() -- threw on null. The entire
    // library view was unreachable. openBrowse() below replaces that proxy.
    browseSection: document.getElementById('browse-section'),
    videoTabs: document.getElementById('video-tabs'),
    browseGrid: document.getElementById('browse-grid'),
    browseLoader: document.getElementById('browse-loader'),
    closeBrowse: document.getElementById('close-browse'),
    modal: document.getElementById('modal'),
    modalImg: document.getElementById('modal-img'),
    modalVideo: document.getElementById('modal-video'),
    modalCaption: document.getElementById('modal-caption'),
    modalClose: document.getElementById('modal-close'),
    modalTitle: document.getElementById('modal-title'),
    modalYear: document.getElementById('modal-year'),
    modalStreaming: document.getElementById('modal-streaming'),
    clipCurrentTime: document.getElementById('clip-current-time'),
    clipTotalTime: document.getElementById('clip-total-time'),
    noMatchesMsg: document.getElementById('no-matches-msg'),
    exploreBtn: document.getElementById('explore-library-btn'),
    uploadForm: document.getElementById('upload-form'),
    // Immersive BG Refs
    modalContainer: document.getElementById('modal-content-container'),
    modalOverlay: document.getElementById('modal-background-overlay')
};

// --- State ---
let browseState = {
    page: 1,
    limit: 20,
    currentTitle: null,
    isLoading: false,
    hasMore: true
};

// --- Drag & Drop Logic ---
const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
};

if (els.form) {
    els.form.addEventListener('dragenter', (e) => {
        handleDrag(e);
        els.dropOverlay.classList.remove('hidden');
        setTimeout(() => els.dropOverlay.classList.remove('opacity-0'), 10);
    });
    els.dropOverlay.addEventListener('dragover', handleDrag);
    els.dropOverlay.addEventListener('dragleave', (e) => {
        handleDrag(e);
        els.dropOverlay.classList.add('opacity-0');
        setTimeout(() => els.dropOverlay.classList.add('hidden'), 200);
    });
    els.dropOverlay.addEventListener('drop', (e) => {
        handleDrag(e);
        els.dropOverlay.classList.add('hidden');
        const files = e.dataTransfer.files;
        if (files.length) handleFile(files[0]);
    });
}

// --- File Handling ---
if (els.dropZone) {
    els.dropZone.addEventListener('click', (e) => {
        console.log("Drop zone clicked", e.target);
        if (e.target !== els.clearBtn && !els.previewContainer.contains(e.target)) {
            if (els.input) {
                console.log("Triggering input click");
                els.input.click();
            } else {
                console.error("Input element not found!");
            }
        }
    });
} else {
    console.error("Drop zone not found!");
}

if (els.input) {
    els.input.addEventListener('change', () => {
        if (els.input.files.length) handleFile(els.input.files[0]);
    });
}

if (els.clearBtn) {
    els.clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetFile();
    });
}

// --- Layout State Management ---

function toggleResultsLayout(show) {
    const uploadCol = document.getElementById('upload-column');
    const resultsCol = document.getElementById('results-column');
    const heroText = document.getElementById('hero-text');
    const dropZone = document.getElementById('drop-zone');
    const wrapper = document.getElementById('content-wrapper');

    if (show) {
        // Switch to Side-by-Side
        // wrapper.classList.remove('flex-col'); // REMOVED: Keep flex-col for mobile stacking!

        if (heroText) heroText.classList.add('hidden'); // Hide big text

        if (uploadCol) {
            // Mobile: Sticky Top, Full Width, Dark BG to cover scrolling results
            uploadCol.classList.remove('items-center'); // Keep w-full for mobile
            uploadCol.classList.add('sticky', 'top-0', 'z-40', 'bg-black/95', 'backdrop-blur-md', 'border-b', 'border-white/10', 'py-4', 'shadow-2xl');

            // Desktop adjustments (overrides)
            uploadCol.classList.add('lg:w-1/4', 'lg:top-8', 'lg:bg-transparent', 'lg:border-none', 'lg:shadow-none', 'lg:py-0', 'lg:backdrop-blur-none');
        }

        if (resultsCol) {
            resultsCol.classList.remove('hidden');
            // Small delay to allow display:flex to apply before opacity transition
            requestAnimationFrame(() => {
                resultsCol.classList.remove('opacity-0');
            });
        }

        // Compact Drop Zone
        if (dropZone) {
            dropZone.classList.add('scale-90', 'h-40'); // Smaller
            dropZone.parentElement.classList.remove('h-screen'); // Remove full height centering
        }

    } else {
        // Reset to Centered
        // wrapper.classList.add('flex-col');
        if (heroText) heroText.classList.remove('hidden');

        if (uploadCol) {
            uploadCol.classList.add('items-center');
            uploadCol.classList.remove('sticky', 'top-0', 'z-40', 'bg-black/95', 'backdrop-blur-md', 'border-b', 'border-white/10', 'py-4', 'shadow-2xl', 'lg:w-1/4', 'lg:top-8', 'lg:bg-transparent', 'lg:border-none', 'lg:shadow-none', 'lg:py-0', 'lg:backdrop-blur-none');
        }

        if (resultsCol) {
            resultsCol.classList.add('hidden', 'opacity-0');
        }

        if (dropZone) {
            dropZone.classList.remove('scale-90', 'h-40');
            // dropZone.parentElement.classList.add('h-screen');
        }
    }
}


// --- UI State Management ---
// Centralized state management to prevent visibility bugs
const UI_STATES = {
    IDLE: 'idle',           // No file, show prompt
    PREVIEW: 'preview',     // File selected, show preview
    SEARCHING: 'searching'  // Search in progress
};

function updateUploadUIState(state) {
    console.log(`UI State Transition: ${state}`);

    switch (state) {
        case UI_STATES.IDLE:
            els.prompt.classList.remove('hidden');
            els.previewContainer.classList.add('hidden');
            els.searchBtn.classList.add('hidden');
            els.searchBtn.disabled = false;
            els.searchBtn.innerText = 'Search Library';
            els.input.value = ''; // Clear input
            if (document.getElementById('text-query')) document.getElementById('text-query').value = '';
            if (document.getElementById('actor-query')) document.getElementById('actor-query').value = '';
            break;

        case UI_STATES.PREVIEW:
            els.prompt.classList.add('hidden');
            els.previewContainer.classList.remove('hidden');
            els.searchBtn.classList.remove('hidden');
            els.searchBtn.disabled = false;
            els.searchBtn.innerText = 'Search Library';
            break;

        case UI_STATES.SEARCHING:
            // Keep preview visible, just update button
            els.searchBtn.disabled = true;
            els.searchBtn.innerText = 'Analyzing...';
            els.statusMsg.classList.add('hidden');
            break;
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/') && !file.type.startsWith('video/')) {
        showStatus('Please upload an image or video file.', 'error');
        return;
    }

    selectedFile = file;
    updateUploadUIState(UI_STATES.PREVIEW);

    if (file.type.startsWith('image/')) {
        // Image Preview
        els.previewImg.src = URL.createObjectURL(file);
        els.previewImg.classList.remove('hidden');
        els.previewVideo.classList.add('hidden');
        els.previewVideo.src = "";
    } else {
        // Video Preview
        els.previewVideo.src = URL.createObjectURL(file);
        els.previewVideo.classList.remove('hidden');
        els.previewImg.classList.add('hidden');
        els.previewImg.src = "";
        els.previewVideo.play();
    }
}

function resetFile() {
    selectedFile = null;
    els.previewImg.src = '';
    els.previewVideo.src = '';

    updateUploadUIState(UI_STATES.IDLE);

    // Reset Layout Global
    toggleResultsLayout(false);
    els.resultsContainer.classList.add('hidden');
    els.browseSection.classList.add('hidden');
    els.statusMsg.classList.add('hidden');

    if (currentSelectedFilename) {
        currentSelectedFilename = null;
    }
}

// --- Video Search Helpers ---

async function extractFrameFromVideo(videoFile, time) {
    return new Promise((resolve) => {
        const video = document.createElement('video');
        video.preload = 'metadata';
        video.src = URL.createObjectURL(videoFile);
        video.muted = true;
        video.playsInline = true;

        video.onloadedmetadata = () => {
            video.currentTime = time;
        };

        video.onseeked = () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob((blob) => {
                URL.revokeObjectURL(video.src);
                resolve(blob);
            }, 'image/jpeg', 0.95);
        };

        video.onerror = () => {
            resolve(null);
        };
    });
}

function getVideoDuration(file) {
    return new Promise((resolve) => {
        const video = document.createElement('video');
        video.preload = 'metadata';
        video.onloadedmetadata = () => {
            URL.revokeObjectURL(video.src);
            resolve(video.duration);
        };
        video.src = URL.createObjectURL(file);
    });
}

function showStatus(message, type = 'info', options = {}) {
    // Clear previous content
    els.statusMsg.innerHTML = '';
    els.statusMsg.classList.remove('hidden', 'text-red-500', 'text-green-500', 'text-yellow-500');

    // Enhanced error UI
    if (type === 'error') {
        els.statusMsg.className = 'mb-8 w-full';
        els.statusMsg.innerHTML = `
            <div class="glass-strong rounded-2xl p-6 border border-red-500/30 max-w-md mx-auto">
                <div class="flex items-start gap-4">
                    <div class="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center shrink-0">
                        <svg class="w-6 h-6 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                        </svg>
                    </div>
                    <div class="flex-1">
                        <h4 class="text-white font-semibold mb-1">${options.title || 'Search Error'}</h4>
                        <p class="text-gray-400 text-sm mb-3">${message}</p>
                        ${options.showRetry ? `
                            <button onclick="retryLastSearch()" class="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 text-sm font-medium rounded-lg transition-colors border border-red-500/30">
                                <span class="flex items-center gap-2">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                                    </svg>
                                    Try Again
                                </span>
                            </button>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    } else if (type === 'success') {
        els.statusMsg.classList.add('text-green-500');
        els.statusMsg.textContent = message;
    } else {
        els.statusMsg.classList.add('text-yellow-500');
        els.statusMsg.textContent = message;
    }
}

// Retry search helper
let lastSearchFile = null;
function retryLastSearch() {
    if (lastSearchFile && selectedFile) {
        els.searchBtn.click();
    } else if (selectedFile) {
        els.searchBtn.click();
    }
}

function displayResults(results) {
    els.resultsGrid.innerHTML = '';
    results.forEach((item, i) => {
        const card = createCard(item, false);
        els.resultsGrid.appendChild(card);
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, i * 100);
    });
}


// --- Search Logic ---

if (els.searchBtn) {
    els.searchBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        e.stopPropagation(); // Stop bubbling to dropZone
        if (!selectedFile) return;

        updateUploadUIState(UI_STATES.SEARCHING); // Use State

        // Switch Layout immediately
        toggleResultsLayout(true);
        els.resultsContainer.classList.remove('hidden');
        els.resultsGrid.innerHTML = ''; // Clear previous

        if (selectedFile.type.startsWith('video/')) {
            await processVideoSearch(selectedFile);
        } else {
            await performImageSearch(selectedFile);
        }
    });
}
// Fallback for form submit if user hits enter
if (els.form) {
    els.form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        updateUploadUIState(UI_STATES.SEARCHING); // Use State

        // Switch Layout immediately
        toggleResultsLayout(true);
        els.resultsContainer.classList.remove('hidden');
        els.resultsGrid.innerHTML = ''; // Clear previous

        if (selectedFile.type.startsWith('video/')) {
            await processVideoSearch(selectedFile);
        } else {
            await performImageSearch(selectedFile);
        }
    });
}

if (els.exploreBtn) {
    els.exploreBtn.addEventListener('click', () => {
        els.noMatchesMsg.classList.add('hidden');
        openBrowse();
    });
}

async function processVideoSearch(file) {
    try {
        const duration = await getVideoDuration(file);
        console.log(`Video duration: ${duration}s`);

        let sampleTimes = [];

        // Strategy: 
        // < 5s: Every 0.5s
        // > 5s: 0.5s, then every 3s
        if (duration < 5) {
            for (let t = 0.5; t < duration; t += 0.5) {
                sampleTimes.push(t);
            }
        } else {
            sampleTimes.push(0.5); // Always check start
            for (let t = 3.5; t < duration; t += 3.0) {
                sampleTimes.push(t);
            }
        }

        if (sampleTimes.length === 0) sampleTimes.push(duration / 2); // Fallback for tiny videos

        console.log(`Sampling frames at: ${sampleTimes}`);
        els.searchBtn.innerText = `Analyzing ${sampleTimes.length} frames...`;

        // Parallel Processing (Chunks of 3)
        let results = [];
        const chunkSize = 3;

        for (let i = 0; i < sampleTimes.length; i += chunkSize) {
            const chunk = sampleTimes.slice(i, i + chunkSize);
            const promises = chunk.map(async (time) => {
                const blob = await extractFrameFromVideo(file, time);
                if (blob) {
                    return await searchFromBlob(blob); // Helper function
                }
                return null;
            });

            const chunkResults = await Promise.all(promises);
            results.push(...chunkResults.filter(r => r !== null));

            // Update Progress
            const progress = Math.min(100, Math.round(((i + chunkSize) / sampleTimes.length) * 100));
            els.searchBtn.innerText = `Analyzing... ${progress}%`;
        }

        // Aggregation Logic
        // Flatten all backend results
        // Structure from backend: { results: [ {filename, score, ...}, ... ] }

        // Aggregation Logic (Group by Video Title)
        // video_title -> { count, bestObj, bestScore }
        // Scores are cosine similarities now, so the best frame for a title is
        // the one with the HIGHEST score, not the lowest distance.
        let aggregation = {};

        results.forEach(res => {
            if (res && res.results) {
                res.results.forEach(item => {
                    const title = item.video_title;
                    const itemScore = typeof item.score === 'number' ? item.score : (item.similarity || 0);

                    if (!aggregation[title]) {
                        aggregation[title] = { count: 0, bestObj: item, bestScore: itemScore };
                    }

                    aggregation[title].count += 1;

                    if (itemScore > aggregation[title].bestScore) {
                        aggregation[title].bestScore = itemScore;
                        aggregation[title].bestObj = item;
                    }
                });
            }
        });

        // Sort Videos:
        // 1. Hit Count (Confidence)
        // 2. Min Distance (Quality)
        let sortedCandidates = Object.values(aggregation).sort((a, b) => {
            if (b.count !== a.count) return b.count - a.count; // Prioritize mostly matched
            return b.bestScore - a.bestScore; // Then best match quality (higher is better)
        });

        // Extract just the best frame object for each video
        const finalResults = sortedCandidates.map(c => {
            // Optional: Add metadata to card about match confidence?
            let bestFrame = c.bestObj;
            if (c.count > 1) {
                // Mark as high confidence if multiple hits
                bestFrame.is_collapsed_match = true;
                bestFrame.match_count = c.count;
            }
            return bestFrame;
        });

        console.log("Aggregated Results:", finalResults);

        if (finalResults.length === 0) {
            showStatus('No matches found in library.', 'error');
            els.noMatchesMsg.classList.remove('hidden');
        } else {
            els.noMatchesMsg.classList.add('hidden');
            displayResults(finalResults);
        }

    } catch (err) {
        console.error("Video processing failed", err);
        showStatus('Failed to process video.', 'error');
    } finally {
        els.searchBtn.disabled = false;
        els.searchBtn.innerText = 'Search Library';
    }
}

// Helper to wrap the fetch for a Blob
async function searchFromBlob(blob) {
    const formData = new FormData();
    formData.append('query_image', blob, 'frame.jpg');

    try {
        const res = await fetch('/api/search', {
            method: 'POST',
            body: formData
        });
        if (!res.ok) return null;
        return await res.json();
    } catch (e) {
        console.error("Frame search error", e);
        return null;
    }
}

// Original Image Search (renamed/wrapped)
async function performImageSearch(file) {
    const formData = new FormData();
    formData.append('query_image', file);

    // Add text query if present
    const textQuery = document.getElementById('text-query')?.value;
    if (textQuery) {
        formData.append('text_query', textQuery);
    }

    // Add actor filter if present
    const actorQuery = document.getElementById('actor-query')?.value;
    if (actorQuery) {
        formData.append('actor_name', actorQuery);
    }

    try {
        const res = await fetch('/api/search', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error('Search failed');

        const data = await res.json();
        console.log("Search Results:", data);

        if (!data.results || data.results.length === 0) {
            showStatus('Try a different screenshot or explore our library.', 'error', {
                title: 'No Matches Found',
                showRetry: false
            });
            els.noMatchesMsg.classList.remove('hidden');
            // Ensure grid is hidden or cleared
            els.resultsGrid.innerHTML = '';
        } else {
            // Success
            els.statusMsg.classList.add('hidden');
            els.noMatchesMsg.classList.add('hidden');
            displayResults(data.results);

            // Check for feedback note (e.g. from prompt)
            if (data.feedback_note) {
                showStatus(data.feedback_note, 'info');
            }
        }
    } catch (err) {
        console.error(err);
        showStatus('Unable to connect to search server. Please check your connection.', 'error', {
            title: 'Connection Error',
            showRetry: true
        });
    } finally {
        els.searchBtn.disabled = false;
        els.searchBtn.innerText = 'Search Library';
    }
}

// --- Helper Functions ---
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// --- Card Component ---
function createCard(data, isBrowseMode) {
    const el = document.createElement('div');
    el.className = `result-card glass rounded-2xl overflow-hidden transform translate-y-4 transition-all duration-300 w-full min-w-[260px]`;

    // Match quality, derived from cosine similarity in [0, 1] where higher is
    // better. The previous version read `data.distance` (an L2 distance, lower
    // is better) and converted it to a percentage with `1 - d / 0.6` -- a
    // hand-picked constant that made the displayed confidence arbitrary. The
    // backend now returns a similarity directly, so the percentage is the
    // score itself.
    const score = typeof data.score === 'number' ? data.score
                : typeof data.similarity === 'number' ? data.similarity
                : null;
    const pct = score === null ? null : Math.round(Math.max(0, Math.min(1, score)) * 100);

    let badgeClass = "bg-gray-600";
    let badgeText = "Match";
    let toneText = "text-gray-400";
    let toneBar = "bg-gray-600";

    if (!isBrowseMode && score !== null) {
        if (score >= 0.95) {
            badgeClass = "bg-gradient-to-r from-emerald-500 to-teal-500"; badgeText = "Perfect";
            toneText = "text-emerald-400"; toneBar = "bg-gradient-to-r from-emerald-500 to-teal-400";
        } else if (score >= 0.90) {
            badgeClass = "bg-gradient-to-r from-green-500 to-emerald-500"; badgeText = "Excellent";
            toneText = "text-green-400"; toneBar = "bg-gradient-to-r from-green-500 to-emerald-400";
        } else if (score >= 0.82) {
            badgeClass = "bg-gradient-to-r from-yellow-500 to-orange-500"; badgeText = "Good";
            toneText = "text-yellow-400"; toneBar = "bg-gradient-to-r from-yellow-500 to-orange-400";
        } else {
            badgeClass = "bg-gray-600"; badgeText = "Possible";
        }
    }

    // Find Video Config for specific URL
    const videoConfig = VIDEO_TITLES.find(v => v.title === data.video_title);
    const youtubeUrl = videoConfig?.streaming_on?.find(s => s.name === 'YouTube')?.url;

    el.innerHTML = `
        <div class="relative aspect-[16/10] bg-gradient-to-br from-gray-900 to-black overflow-hidden group cursor-pointer card-image-container"
             data-filename="${data.filename}" 
             data-video-title="${data.video_title}" 
             data-time-sec="${data.time_sec}">
            <img src="/output_frames/${data.filename}" 
                 class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" 
                 loading="lazy">
            
            <!-- Gradient overlay on hover -->
            <div class="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            
            <!-- Play icon on hover -->
            <div class="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none">
                <div class="w-16 h-16 rounded-full bg-purple-500/90 flex items-center justify-center backdrop-blur-sm shadow-xl">
                    <svg class="w-8 h-8 text-white ml-1" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z"/>
                    </svg>
                </div>
            </div>

            <!-- YouTube Link (Top Right) -->
            ${youtubeUrl ? `
            <a href="${youtubeUrl}" target="_blank" onclick="event.stopPropagation();" 
               class="absolute top-3 right-3 z-20 bg-red-600 hover:bg-red-700 text-white p-2 rounded-full shadow-lg opacity-0 group-hover:opacity-100 transition-all transform translate-y-2 group-hover:translate-y-0" title="Watch full video on YouTube">
                <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/>
                </svg>
            </a>` : ''}
            
            <!-- Badges -->
            ${data.is_collapsed_match ? `<div class="absolute top-3 left-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-xs font-bold px-3 py-1 rounded-full shadow-lg backdrop-blur-sm flex items-center gap-1 z-10 animate-pulse">
                <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                ${data.match_count} Matches
            </div>` : ''}

            ${!data.is_collapsed_match && data.feedback_note ? `<div class="absolute top-3 left-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white text-xs font-bold px-3 py-1 rounded-full shadow-lg backdrop-blur-sm flex items-center gap-1 z-10">
                <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path></svg>
                ${data.feedback_note}
            </div>` : ''}
            
            ${!data.is_collapsed_match && !isBrowseMode && !data.feedback_note ? `<div class="absolute top-3 left-3 ${badgeClass} text-white text-xs font-bold px-3 py-1 rounded-full shadow-lg backdrop-blur-sm z-10">
                ${badgeText}
            </div>` : ''}
        </div>

        <div class="p-5 space-y-3">
            <div class="flex items-center justify-between">
                <h3 class="text-lg font-bold text-white truncate max-w-[70%]">${data.video_title}</h3>
                ${youtubeUrl ? `
                <a href="${youtubeUrl}" target="_blank" onclick="event.stopPropagation()" class="text-xs text-gray-500 hover:text-red-500 flex items-center gap-1 transition-colors">
                    <span class="truncate max-w-[100px]">Watch Source</span>
                    <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
                </a>` : ''}
            </div>
            <p class="text-gray-400 text-sm font-mono flex items-center gap-2">
                <span>⏱ ${formatTime(data.time_sec)}</span>
            </p>
            ${!isBrowseMode ? `
                <!-- Confidence, straight from the cosine similarity -->
                <div class="space-y-1">
                    <div class="flex items-center justify-between text-xs">
                        <span class="text-gray-500">Match Confidence</span>
                        <span class="font-semibold ${toneText}">${pct === null ? '--' : pct + '%'}</span>
                    </div>
                    <div class="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                        <div class="h-full rounded-full transition-all duration-500 ${toneBar}"
                             style="width: ${pct === null ? 0 : Math.max(4, pct)}%"></div>
                    </div>
                </div>
                ${data.is_boosted ? '<div class="text-xs text-purple-400 flex items-center gap-1"><svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path></svg> Previously Selected</div>' : ''}
                <button class="select-btn w-full py-2.5 rounded-xl glass-strong hover:bg-purple-500/20 text-gray-300 hover:text-white text-sm font-semibold transition-all border border-transparent hover:border-purple-500/50"
                        onclick="event.stopPropagation(); submitFeedback('${data.filename}', this.closest('.result-card'))">
                    Select Scene
                </button>
            ` : ''}
        </div>
    `;

    // Add click event listener to the image container
    const imageContainer = el.querySelector('.card-image-container');
    if (imageContainer) {
        imageContainer.addEventListener('click', () => {
            const filename = imageContainer.dataset.filename;
            const videoTitle = imageContainer.dataset.videoTitle;
            const timeSec = parseFloat(imageContainer.dataset.timeSec);
            console.log("Card clicked:", filename, videoTitle, timeSec);
            openModal(filename, videoTitle, timeSec, data.detected_actors);
        });
    }

    return el;
}

// --- Feedback Submission ---
let currentSelectedFilename = null;

async function submitFeedback(filename, cardElement) {
    console.log(`Feedback action: ${filename}`);

    // Toggle Logic
    if (currentSelectedFilename === filename) {
        // Already selected -> UNSELECT (Exit Focus Mode)
        await exitFocusMode(cardElement, filename);
    } else {
        // New Selection -> SELECT (Enter Focus Mode)
        // Note: In Focus Mode, others are hidden, so you can't click them directly.
        // You must unselect first. But safe to handle switch just in case.
        if (currentSelectedFilename) {
            // Should technically unlock first, but let's just trigger new focus
            // This case is rare if others are hidden.
            await sendFeedbackAction(currentSelectedFilename, 'remove');
        }
        await enterFocusMode(cardElement, filename);
    }
}

async function enterFocusMode(card, filename) {
    console.log("Entering Focus Mode");
    currentSelectedFilename = filename;

    // 1. Send Feedback
    sendFeedbackAction(filename, 'add');

    // 2. Hide Upload Form (Visual Cleanup)
    if (els.uploadForm) {
        els.uploadForm.style.transition = 'opacity 0.5s ease-out';
        els.uploadForm.style.opacity = '0';
        setTimeout(() => els.uploadForm.classList.add('hidden'), 500);
    }

    // 3. Visuals: Fade out OTHERS
    const allCards = document.querySelectorAll('.result-card');
    allCards.forEach(c => {
        if (c !== card) {
            c.style.transition = 'all 0.5s ease-out';
            c.style.opacity = '0';
            c.style.transform = 'scale(0.9)';
            setTimeout(() => c.style.display = 'none', 500);
        }
    });

    // 4. Center Grid
    setTimeout(() => {
        els.resultsGrid.style.display = 'flex';
        els.resultsGrid.style.justifyContent = 'center';
        els.resultsGrid.style.alignItems = 'flex-start';
        els.resultsGrid.style.minHeight = '80vh';

        // 5. Enlarge Selection
        card.classList.add('card-selected');
        card.style.transition = 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)';
        card.style.transform = 'scale(1.2)'; // Scale up
        card.style.maxWidth = '600px';
        card.style.zIndex = '50';
        card.style.margin = '0 auto';

        // 6. Update Button State
        const btn = card.querySelector('.select-btn');
        btn.textContent = "✕ Unselect / Back";
        btn.className = "select-btn w-full py-3 rounded-xl bg-gray-700 hover:bg-gray-600 text-white font-bold shadow-lg transition-all border border-gray-500";

        // 7. Show "New Search" / "Home" Options
        showNewSearchButton();
    }, 500);
}

async function exitFocusMode(card, filename) {
    console.log("Exiting Focus Mode");
    currentSelectedFilename = null;

    // 1. Remove Feedback
    sendFeedbackAction(filename, 'remove');

    // 2. Reset Grid
    els.resultsGrid.style.display = 'grid'; // Back to grid
    els.resultsGrid.style.justifyContent = '';
    els.resultsGrid.style.alignItems = '';
    els.resultsGrid.style.minHeight = '';

    // 3. Reset Card Styles
    card.classList.remove('card-selected');
    card.style.transform = 'translateY(0)';
    card.style.maxWidth = '';
    card.style.zIndex = '';
    card.style.margin = '';

    const btn = card.querySelector('.select-btn');
    btn.textContent = "Select Scene";
    btn.className = "select-btn w-full py-2.5 rounded-xl glass-strong hover:bg-purple-500/20 text-gray-300 hover:text-white text-sm font-semibold transition-all border border-transparent hover:border-purple-500/50";

    // 4. Show Others
    const allCards = document.querySelectorAll('.result-card');
    allCards.forEach(c => {
        c.style.display = ''; // Reset display
        setTimeout(() => {
            c.style.opacity = '1';
            c.style.transform = 'translateY(0)';
        }, 50);
    });

    // 5. Restore Upload Form
    if (els.uploadForm) {
        els.uploadForm.classList.remove('hidden');
        setTimeout(() => els.uploadForm.style.opacity = '1', 50);
    }

    // 6. Hide New Search Button
    const nsBtn = document.getElementById('new-search-btn');
    if (nsBtn) nsBtn.remove();
}

async function sendFeedbackAction(filename, action) {
    try {
        await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                selected_filename: filename,
                query_hash: currentQueryHash,
                action: action
            })
        });
    } catch (e) {
        console.warn("Feedback failed:", e);
    }
}

function showNewSearchButton() {
    // Avoid duplicates
    if (document.getElementById('new-search-btn')) return;

    let newSearchBtn = document.createElement('div');
    newSearchBtn.id = 'new-search-btn';
    newSearchBtn.className = 'text-center mt-12 animate-fade-in-up w-full absolute bottom-[-100px] left-0';
    newSearchBtn.innerHTML = `
        <button class="btn-primary px-8 py-4 rounded-full font-bold text-lg inline-flex items-center gap-3 shadow-2xl shadow-purple-500/20">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
            </svg>
            New Search
        </button>
    `;

    // Append to wrapper or grid? Grid is flex col now.
    els.resultsGrid.appendChild(newSearchBtn);

    newSearchBtn.querySelector('button').addEventListener('click', () => {
        location.reload();
    });
}
// --- Browse Logic ---
function openBrowse() {
    if (!els.browseSection) return;
    els.browseSection.classList.remove('hidden');
    renderTabs();            // tabs for manual filtering
    loadExploreDashboard();  // history + random picks
    els.browseSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
window.openBrowse = openBrowse;

// Infinite Scroll Listeners (Only active when NOT in Dashboard mode, or customize)
window.addEventListener('scroll', () => {
    if (els.browseSection.classList.contains('hidden')) return;
    if (browseState.isDashboard) return; // Don't infinite scroll dashboard

    // Check if near bottom
    if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 500) {
        if (browseState.hasMore && !browseState.isLoading) {
            loadBrowseData(browseState.currentTitle, false);
        }
    }
});

async function loadExploreDashboard() {
    browseState.isDashboard = true;
    els.browseGrid.innerHTML = '';
    els.browseLoader.classList.remove('hidden');

    try {
        const res = await fetch('/api/explore');
        const data = await res.json();
        els.browseLoader.classList.add('hidden');

        // 1. History Section
        if (data.history && data.history.length > 0) {
            const historyTitle = document.createElement('h3');
            historyTitle.className = "col-span-full text-2xl font-bold text-white mt-4 mb-2 flex items-center gap-2";
            historyTitle.innerHTML = `<span class="text-purple-400">⚡</span> Recently Discovered`;
            els.browseGrid.appendChild(historyTitle);

            data.history.forEach(item => {
                const card = createCard(item, true);
                card.classList.add('animate-fade-in-up');
                els.browseGrid.appendChild(card);
            });
        }

        // 2. Random Section
        const randomTitle = document.createElement('h3');
        randomTitle.className = "col-span-full text-2xl font-bold text-white mt-12 mb-2 flex items-center gap-2";
        randomTitle.innerHTML = `<span class="text-blue-400">🎲</span> Random Picks`;
        els.browseGrid.appendChild(randomTitle);

        data.random.forEach(item => {
            const card = createCard(item, true);
            card.classList.add('animate-fade-in-up');
            els.browseGrid.appendChild(card);
        });

    } catch (e) {
        console.error("Explore dashboard failed:", e);
        els.browseLoader.classList.add('hidden');
    }
}

if (els.closeBrowse) {
    els.closeBrowse.addEventListener('click', () => {
        els.browseSection.classList.add('hidden');
    });
}

function renderTabs() {
    els.videoTabs.innerHTML = '';
    VIDEO_TITLES.forEach((vidObj, idx) => {
        const title = vidObj.title;
        const btn = document.createElement('button');
        btn.textContent = title;
        btn.className = `browse-tab px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap ${idx === 0 ? 'active' : 'bg-gray-800 text-gray-400'}`;
        btn.onclick = () => {
            document.querySelectorAll('.browse-tab').forEach(b => {
                b.classList.remove('active', 'bg-cyan-600', 'text-white');
                b.classList.add('bg-gray-800', 'text-gray-400');
            });
            btn.className = "browse-tab px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap active";
            loadBrowseData(title, true);
        };
        els.videoTabs.appendChild(btn);
    });
}

async function loadBrowseData(title, reset = false) {
    if (reset) {
        browseState.isDashboard = false; // Disable dashboard mode
        browseState.page = 1;
        browseState.hasMore = true;
        browseState.currentTitle = title;
        els.browseGrid.innerHTML = '';
    }

    if (!browseState.hasMore) return;

    browseState.isLoading = true;
    els.browseLoader.classList.remove('hidden');

    try {
        const res = await fetch(`/api/all_frames?video_title=${title}&page=${browseState.page}&limit=${browseState.limit}`);
        const data = await res.json();

        browseState.isLoading = false;
        els.browseLoader.classList.add('hidden');

        if (data.frames.length < browseState.limit) {
            browseState.hasMore = false;
        } else {
            browseState.page++;
        }

        data.frames.forEach((f, idx) => {
            // Browse mode shows no confidence, so no score is passed.
            const card = createCard(f, true);
            card.style.opacity = 1;
            card.style.transform = 'none';
            els.browseGrid.appendChild(card);
        });
    } catch (err) {
        console.error(err);
        browseState.isLoading = false;
        els.browseLoader.classList.add('hidden');
    }
}

// --- Modal & Custom Player Logic ---
let videoLoopStart = 0;
let videoLoopEnd = 0;
let isVideoPlaying = false;
let animationFrameId = null;

// Helper function for time formatting
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

const openModal = async (filename, videoTitle, timeSec, detectedActors = null) => {
    console.log("openModal called:", filename, videoTitle, timeSec);

    // Check if modal elements exist
    if (!els.modal || !els.modalImg) return;

    // 1. Reset & Show Modal
    els.modalImg.src = `/output_frames/${filename}`;
    els.modalImg.classList.remove('hidden');

    // Show the Container
    els.modal.classList.remove('hidden');
    // Small delay for fade connection
    setTimeout(() => els.modal.classList.remove('opacity-0'), 10);

    if (els.modalVideo) {
        els.modalVideo.classList.add('hidden');
        els.modalVideo.pause();
        els.modalVideo.src = "";
    }

    // Set Immersive Background
    if (els.modalContainer) {
        els.modalContainer.style.backgroundImage = `url('/output_frames/${filename}')`;
        els.modalContainer.classList.remove('bg-[#0b0c15]'); // Remove solid background class
        els.modalContainer.style.backgroundColor = 'transparent'; // Force transparent
        els.modalContainer.classList.add('modal-immersive-bg', 'modal-immersive-active');
    }

    // Reset Custom Controls
    const progressFill = document.getElementById('progress-fill');
    const playBtnIcon = document.querySelector('#play-pause-btn svg path');
    const videoLoader = document.getElementById('video-loader');
    if (progressFill) progressFill.style.width = '0%';
    if (playBtnIcon) playBtnIcon.setAttribute('d', 'M8 5v14l11-7z'); // Play icon
    if (videoLoader) videoLoader.style.opacity = '1';

    if (els.modalTitle) els.modalTitle.textContent = videoTitle;

    // Find video object metadata
    const videoObj = VIDEO_TITLES.find(v => v.title === videoTitle) || {};

    if (els.modalYear) els.modalYear.textContent = videoObj.year || '—';

    // Streaming Badges
    if (els.modalStreaming) {
        els.modalStreaming.innerHTML = '';
        if (videoObj.streaming_on) {
            // Separate YouTube from others for emphasis
            const youtubeLink = videoObj.streaming_on.find(s => s.name === 'YouTube');
            const others = videoObj.streaming_on.filter(s => s.name !== 'YouTube');

            // 1. Primary YouTube Button
            if (youtubeLink) {
                const ytBtn = document.createElement('a');
                ytBtn.href = youtubeLink.url;
                ytBtn.target = "_blank";
                ytBtn.className = 'px-4 py-2 bg-red-600/90 hover:bg-red-500 backdrop-blur-sm text-white text-xs font-bold rounded-lg transition-all shadow-lg shadow-red-900/20 hover:shadow-red-600/40 flex items-center gap-2 group';
                ytBtn.innerHTML = `
                        <svg class="w-4 h-4 fill-current" viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/></svg>
                        Watch
                    `;
                els.modalStreaming.appendChild(ytBtn);
            }

            // 2. Secondary Badges
            const othersContainer = document.createElement('div');
            othersContainer.className = 'flex flex-wrap gap-2';

            others.forEach(platform => {
                const badge = document.createElement('a');
                badge.href = platform.url;
                badge.target = "_blank";
                badge.className = 'px-3 py-1 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-xs text-gray-400 hover:text-white transition-colors flex items-center gap-1';
                badge.innerHTML = `${platform.name} <svg class="w-3 h-3 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6m4-2h6m0 0v6m0-6L10 14"></path></svg>`;
                othersContainer.appendChild(badge);
            });
            els.modalStreaming.appendChild(othersContainer);
        }
    }

    // Cast List
    const modalCastGrid = document.getElementById('modal-cast-grid');
    if (modalCastGrid) {
        modalCastGrid.innerHTML = '';

        // Priority: Detected Scene Actors > Movie Cast
        let actorsToShow = [];
        let isSceneSpecific = false;

        if (detectedActors && detectedActors.length > 0) {
            actorsToShow = detectedActors;
            isSceneSpecific = true;
        } else if (videoObj.actors) {
            actorsToShow = videoObj.actors;
        }

        if (actorsToShow.length > 0) {

            // Optional header to indicate source
            if (isSceneSpecific) {
                const header = document.createElement('div');
                header.className = 'col-span-full text-xs text-emerald-400 font-bold mb-1 uppercase tracking-wider opacity-80';
                header.textContent = 'Detected in Scene';
                modalCastGrid.appendChild(header);
            }

            actorsToShow.forEach(actor => {
                const actorRow = document.createElement('div');
                actorRow.className = 'flex items-center gap-3 p-2 rounded-lg hover:bg-white/5 transition-colors';
                actorRow.innerHTML = `
                        <img src="${actor.image}" class="w-10 h-10 rounded-full object-cover bg-gray-800">
                        <div>
                            <div class="text-white text-sm font-semibold leading-none mb-1">${actor.name}</div>
                            <div class="text-purple-400 text-xs leading-none">${actor.character}</div>
                        </div>
                    `;
                modalCastGrid.appendChild(actorRow);
            });
        }
    }


    // 3. Setup Video Player (Custom Clip Logic)
    if (els.modalVideo) {
        const start = Math.max(0, timeSec - 1);
        const end = start + 6; // 6 second clip

        videoLoopStart = start;
        videoLoopEnd = end;
        isVideoPlaying = false;

        // Load src without fragment first to ensure metadata loads
        // els.modalVideo.currentTime = start; // MOVED TO onloadedmetadata


        // --- Custom Player Handlers ---

        // Time Update (Progress Bar & Loop)
        els.modalVideo.ontimeupdate = () => {
            const current = els.modalVideo.currentTime;

            // Loop Logic: If we hit end of clip, restart
            if (current >= videoLoopEnd) {
                els.modalVideo.currentTime = videoLoopStart;
                // Optional: Pause on end vs Loop? Let's Loop.
            }

            // Progress Update
            const totalDuration = videoLoopEnd - videoLoopStart;
            const progress = Math.max(0, current - videoLoopStart);
            const percent = Math.min(100, (progress / totalDuration) * 100);

            if (progressFill) progressFill.style.width = `${percent}%`;

            // Time Text Update
            const curTimeEl = document.getElementById('clip-current-time');
            const totTimeEl = document.getElementById('clip-total-time');
            if (curTimeEl) curTimeEl.textContent = formatTime(progress);
            if (totTimeEl) totTimeEl.textContent = formatTime(totalDuration);
        };

        // New: onloadedmetadata handler
        els.modalVideo.onloadedmetadata = () => {
            els.modalVideo.currentTime = start;
        };

        // Loading State
        els.modalVideo.oncanplay = () => {
            if (videoLoader) videoLoader.style.opacity = '0';
            els.modalVideo.classList.remove('hidden');
            els.modalImg.classList.add('hidden');
            // Auto-play
            togglePlay(true);
        };

        // Fallback: Ensure visible if playing (fixes "audio no video" if oncanplay missed)
        els.modalVideo.onplay = () => {
            els.modalVideo.classList.remove('hidden');
            els.modalImg.classList.add('hidden');
            if (videoLoader) videoLoader.style.opacity = '0';
        };

        // Error handler: If video fails to load, show message and keep image
        els.modalVideo.onerror = () => {
            console.warn("Video failed to load, showing image fallback");
            if (videoLoader) videoLoader.style.opacity = '0';
            els.modalVideo.classList.add('hidden');
            els.modalImg.classList.remove('hidden');
            // Show a toast/message
            const modalCaption = document.getElementById('modal-caption');
            if (modalCaption && !modalCaption.dataset.errorShown) {
                modalCaption.innerHTML = `<span class="text-yellow-400">⚠️ Video unavailable</span> - Showing frame preview. ${modalCaption.innerHTML}`;
                modalCaption.dataset.errorShown = 'true';
            }
        };

        // Play/Pause Function
        const togglePlay = (forcePlay = null) => {
            if (forcePlay === true || els.modalVideo.paused) {
                els.modalVideo.play().then(() => {
                    isVideoPlaying = true;
                    if (playBtnIcon) playBtnIcon.setAttribute('d', 'M6 4h4v16H6V4zm8 0h4v16h-4V4z'); // Pause icon
                }).catch(e => console.log("Autoplay blocked:", e));
            } else {
                els.modalVideo.pause();
                isVideoPlaying = false;
                if (playBtnIcon) playBtnIcon.setAttribute('d', 'M8 5v14l11-7z'); // Play icon
            }
        };

        // Bind Controls
        const playBtn = document.getElementById('play-pause-btn');
        if (playBtn) playBtn.onclick = (e) => { e.stopPropagation(); togglePlay(); };

        const videoContainer = document.querySelector('.video-container'); // Click handling on container
        if (videoContainer) videoContainer.onclick = () => togglePlay();

        // Trigger Load AFTER handlers are set (Fixes race condition)
        els.modalVideo.src = `/api/video/${videoObj.filename}`;
    }

    // 4. AI Description
    els.modalCaption.innerText = "Analyzing scene context...";
    els.modalCaption.classList.add('animate-pulse');

    try {
        const res = await fetch(`/api/caption?filename=${filename}`);
        const data = await res.json();
        els.modalCaption.classList.remove('animate-pulse');

        // Typewriter effect
        els.modalCaption.innerText = "";
        const text = data.description;
        let i = 0;
        function typeWriter() {
            if (i < text.length) {
                els.modalCaption.innerHTML += text.charAt(i);
                i++;
                setTimeout(typeWriter, 15); // Fast typing
            }
        }
        typeWriter();
    } catch (e) {
        els.modalCaption.innerText = "Description unavailable.";
        els.modalCaption.classList.remove('animate-pulse');
    }
};

// Make openModal globally accessible
window.openModal = openModal;

const closeModal = () => {
    els.modal.classList.add('opacity-0');
    setTimeout(() => {
        els.modal.classList.add('hidden');
        els.modalVideo.pause();
        els.modalVideo.src = ""; // Stop buffering

        // Reset Immersive Background
        if (els.modalContainer) {
            els.modalContainer.style.backgroundImage = '';
            els.modalContainer.classList.remove('modal-immersive-bg', 'modal-immersive-active');
        }
    }, 300);
};

if (els.modalClose) {
    els.modalClose.addEventListener('click', closeModal);
}
if (els.modal) {
    els.modal.addEventListener('click', (e) => {
        if (e.target === els.modal) closeModal();
    });
}
