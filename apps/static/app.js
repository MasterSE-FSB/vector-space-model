document.addEventListener('DOMContentLoaded', () => {
    // State management
    let activeTab = 'semantic';
    let availableModels = [];
    let activeModel = '';

    // DOM Elements
    const navItems = document.querySelectorAll('.nav-item');
    const searchForm = document.getElementById('search-form');
    const searchQueryInput = document.getElementById('search-query');
    const searchModelSelect = document.getElementById('search-model');
    const modelSelectGroup = document.getElementById('model-select-group');
    const topKInput = document.getElementById('top-k');
    const topKVal = document.getElementById('top-k-val');
    
    // Status Elements
    const milvusDot = document.getElementById('milvus-status-dot');
    const milvusText = document.getElementById('milvus-status-text');
    const statusDetails = document.getElementById('status-details');

    // Views
    const idleView = document.getElementById('idle-view');
    const loadingView = document.getElementById('loading-view');
    const errorView = document.getElementById('error-view');
    const errorMessage = document.getElementById('error-message');
    const singleResultsView = document.getElementById('single-results-view');
    const resultsInfo = document.getElementById('results-info');
    const resultsGrid = document.getElementById('results-grid');
    const compareResultsView = document.getElementById('compare-results-view');
    const compareInfo = document.getElementById('compare-info');
    const compareColumnsContainer = document.getElementById('compare-columns-container');

    // Init UI updates
    topKInput.addEventListener('input', (e) => {
        topKVal.textContent = e.target.value;
    });

    // Navigation Tab Switching
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            activeTab = item.dataset.tab;
            
            // Adjust form and configs based on tab
            updateUIForTab();
        });
    });

    function updateUIForTab() {
        if (activeTab === 'semantic') {
            modelSelectGroup.classList.remove('hidden');
            searchQueryInput.placeholder = "Nhập câu mô tả ngữ nghĩa (ví dụ: 'fantasy adventure with magic')...";
        } else if (activeTab === 'tfidf') {
            modelSelectGroup.classList.add('hidden');
            searchQueryInput.placeholder = "Nhập từ khóa tìm kiếm baseline (ví dụ: 'Naruto Shippuden')...";
        } else if (activeTab === 'similar') {
            modelSelectGroup.classList.remove('hidden');
            searchQueryInput.placeholder = "Nhập chính xác tên Anime để tìm phim tương tự (ví dụ: 'Death Note')...";
        } else if (activeTab === 'compare') {
            modelSelectGroup.classList.add('hidden'); // Compare comparisons do all models automatically
            searchQueryInput.placeholder = "Nhập câu truy vấn so sánh kết quả giữa các models...";
        }
    }

    // Fetch dynamic models list and system status
    async function initSystem() {
        try {
            // Fetch root metadata
            const metaRes = await fetch('/');
            if (!metaRes.ok) throw new Error("Không thể kết nối API");
            const metaData = await metaRes.json();

            // Render status
            if (metaData.milvus_connected) {
                milvusDot.className = 'status-indicator online';
                milvusText.textContent = `Milvus: Connected`;
            } else {
                milvusDot.className = 'status-indicator offline';
                milvusText.textContent = `Milvus: Disconnected`;
            }

            statusDetails.innerHTML = '';
            if (metaData.models) {
                metaData.models.forEach(m => {
                    const cnt = metaData.vectors && metaData.vectors[m.key] !== undefined ? metaData.vectors[m.key] : '?';
                    const div = document.createElement('div');
                    div.innerHTML = `• <strong>${m.key}</strong>: ${cnt} vectors`;
                    statusDetails.appendChild(div);
                });
            }

            // Fetch models
            const modelsRes = await fetch('/models');
            if (modelsRes.ok) {
                const modelsData = await modelsRes.json();
                availableModels = Object.keys(modelsData);
                
                // Populate select
                searchModelSelect.innerHTML = '';
                availableModels.forEach((m, idx) => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = `${m} (${modelsData[m].hf_name.split('/').pop()})`;
                    if (idx === 0) opt.selected = true;
                    searchModelSelect.appendChild(opt);
                });
                
                activeModel = searchModelSelect.value;
            }
        } catch (err) {
            console.error(err);
            milvusDot.className = 'status-indicator offline';
            milvusText.textContent = `API Offline`;
            statusDetails.innerHTML = '<div>Không kết nối được server backend. Hãy kiểm tra console.</div>';
        }
    }

    searchModelSelect.addEventListener('change', (e) => {
        activeModel = e.target.value;
    });

    // Form submit search
    searchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = searchQueryInput.value.trim();
        if (!query) return;

        showState('loading');

        try {
            const topK = parseInt(topKInput.value, 10);
            
            if (activeTab === 'semantic') {
                const res = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, model: activeModel, top_k: topK })
                });
                if (!res.ok) throw new Error(await getErrorMessage(res));
                const data = await res.json();
                renderSingleResults(data.results, `Kết quả tìm kiếm ngữ nghĩa cho "${query}" bằng model ${data.model}`);
            } 
            else if (activeTab === 'tfidf') {
                const res = await fetch('/search/tfidf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, top_k: topK })
                });
                if (!res.ok) throw new Error(await getErrorMessage(res));
                const data = await res.json();
                renderSingleResults(data.results, `Kết quả tìm kiếm TF-IDF baseline cho "${query}"`);
            } 
            else if (activeTab === 'similar') {
                const res = await fetch('/search/similar', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ anime_name: query, model: activeModel, top_k: topK })
                });
                if (!res.ok) throw new Error(await getErrorMessage(res));
                const data = await res.json();
                renderSingleResults(data.results, `Anime tương tự như "${query}" (Model: ${data.model})`);
            } 
            else if (activeTab === 'compare') {
                // Call semantic compare endpoint
                const compareRes = await fetch('/search/compare', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, top_k: topK })
                });
                if (!compareRes.ok) throw new Error(await getErrorMessage(compareRes));
                const compareData = await compareRes.json();

                // Let's also fetch TF-IDF results in parallel for an extra comparison point!
                let tfidfResults = [];
                try {
                    const tfidfRes = await fetch('/search/tfidf', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, top_k: topK })
                    });
                    if (tfidfRes.ok) {
                        const tfData = await tfidfRes.json();
                        tfidfResults = tfData.results;
                    }
                } catch (err) {
                    console.warn("TF-IDF baseline fetch error during comparison:", err);
                }

                renderCompareResults(compareData.results, tfidfResults, query);
            }
        } catch (err) {
            showState('error', err.message);
        }
    });

    async function getErrorMessage(response) {
        try {
            const detail = await response.json();
            return detail.detail || "Đã xảy ra lỗi không xác định.";
        } catch (e) {
            return `Lỗi HTTP ${response.status}: ${response.statusText}`;
        }
    }

    function showState(state, message = '') {
        idleView.classList.add('hidden');
        loadingView.classList.add('hidden');
        errorView.classList.add('hidden');
        singleResultsView.classList.add('hidden');
        compareResultsView.classList.add('hidden');

        if (state === 'idle') {
            idleView.classList.remove('hidden');
        } else if (state === 'loading') {
            loadingView.classList.remove('hidden');
        } else if (state === 'error') {
            errorView.classList.remove('hidden');
            errorMessage.textContent = message;
        } else if (state === 'single') {
            singleResultsView.classList.remove('hidden');
        } else if (state === 'compare') {
            compareResultsView.classList.remove('hidden');
        }
    }

    // Format number helper
    function formatMembers(num) {
        if (num === null || num === undefined) return '0';
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toLocaleString();
    }

    // Helper to request Similar search
    window.findSimilar = function(animeName) {
        searchQueryInput.value = animeName;
        // switch tab
        const simTab = document.querySelector('[data-tab="similar"]');
        if (simTab) simTab.click();
        // Trigger submit
        searchForm.dispatchEvent(new Event('submit'));
    };

    function renderSingleResults(hits, infoText) {
        showState('single');
        resultsInfo.textContent = infoText;
        resultsGrid.innerHTML = '';

        if (!hits || hits.length === 0) {
            resultsGrid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 3rem;">Không tìm thấy anime nào phù hợp.</div>';
            return;
        }

        hits.forEach((hit, idx) => {
            const card = document.createElement('div');
            card.className = 'anime-card';
            card.style.animationDelay = `${idx * 0.05}s`;

            // Genre pills
            const genres = hit.genre ? hit.genre.split(',').map(g => g.trim()) : [];
            const genreHtml = genres.slice(0, 4).map(g => `<span class="genre-pill">${g}</span>`).join('');

            // Star Rating
            const rating = hit.rating !== null ? hit.rating.toFixed(1) : 'N/A';
            
            // Score Display
            const scoreLabel = hit.score >= 0 && hit.score <= 1.05 ? `Độ tương đồng: ${(hit.score * 100).toFixed(1)}%` : `Score: ${hit.score.toFixed(3)}`;

            card.innerHTML = `
                <div class="card-header">
                    <h3 class="anime-title" title="${hit.name}">${hit.name}</h3>
                    <span class="anime-score">${scoreLabel}</span>
                </div>
                <div class="anime-genres">
                    ${genreHtml}
                </div>
                <div class="card-footer">
                    <div class="card-meta-left">
                        <span class="anime-type">${hit.type || 'N/A'}</span>
                        <span class="anime-rating">
                            <span class="anime-rating-star">⭐</span> ${rating}
                        </span>
                    </div>
                    <span class="anime-members">👥 ${formatMembers(hit.members)}</span>
                </div>
                <button class="similar-btn" onclick="findSimilar('${hit.name.replace(/'/g, "\\'")}')">🌀 Tìm tương tự</button>
            `;
            resultsGrid.appendChild(card);
        });
    }

    function renderCompareResults(compareMap, tfidfList, query) {
        showState('compare');
        compareInfo.textContent = `So sánh song song cho câu truy vấn: "${query}"`;
        compareColumnsContainer.innerHTML = '';

        // Add semantic models
        Object.entries(compareMap).forEach(([modelName, hits]) => {
            createCompareColumn(modelName, hits);
        });

        // Add TF-IDF baseline column if available
        if (tfidfList && tfidfList.length > 0) {
            createCompareColumn('TF-IDF Baseline', tfidfList);
        }
    }

    function createCompareColumn(title, hits) {
        const col = document.createElement('div');
        col.className = 'compare-column';
        
        col.innerHTML = `
            <div class="column-header">
                <div class="column-title">
                    <span>⚡</span> ${title}
                </div>
                <div class="column-badge">${hits.length} items</div>
            </div>
            <div class="column-results"></div>
        `;

        const resultsContainer = col.querySelector('.column-results');

        if (!hits || hits.length === 0) {
            resultsContainer.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 2rem;">No results.</div>';
        } else {
            hits.forEach((hit, idx) => {
                const item = document.createElement('div');
                item.className = 'compare-hit-item';
                
                // Rating label
                const rating = hit.rating !== null ? hit.rating.toFixed(1) : 'N/A';
                // Score representation
                const scoreLabel = hit.score >= 0 && hit.score <= 1.05 ? `${(hit.score * 100).toFixed(0)}%` : hit.score.toFixed(2);

                item.innerHTML = `
                    <div class="hit-rank">#${idx + 1}</div>
                    <div class="hit-body">
                        <div class="hit-name" title="${hit.name}">${hit.name}</div>
                        <div class="hit-meta">
                            <span>${hit.type || 'N/A'}</span>
                            <span style="color: #f59e0b;">⭐ ${rating}</span>
                            <span class="hit-score">Score: ${scoreLabel}</span>
                        </div>
                    </div>
                `;
                resultsContainer.appendChild(item);
            });
        }

        compareColumnsContainer.appendChild(col);
    }

    // Run system init
    initSystem();
});
