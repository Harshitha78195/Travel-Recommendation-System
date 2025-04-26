document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Handle recommendation form submission
    const recommendationForm = document.getElementById('recommendation-form');
    if (recommendationForm) {
        recommendationForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const queryInput = document.getElementById('query');
            const query = queryInput.value.trim();
            
            if (!query) {
                showError('Please enter your travel request');
                return;
            }

            // Show loading state
            const submitBtn = document.querySelector('#recommendation-form button[type="submit"]');
            const originalBtnText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            submitBtn.disabled = true;
            
            // Hide any previous errors
            hideError();
            
            // Show global loading spinner
            document.getElementById('loadingSpinner').style.display = 'flex';

            // Send AJAX request to backend
            fetch('/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: new URLSearchParams({
                    query: query
                })
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errData => {
                        throw new Error(errData.error || 'Network response was not ok');
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Display the query analysis
                displayQueryAnalysis(data);
                
                // Display the recommendations
                displayRecommendations(data);
            })
            .catch(error => {
                console.error('Error:', error);
                showError(error.message);
            })
            .finally(() => {
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
                document.getElementById('loadingSpinner').style.display = 'none';
            });
        });
    }
     
    // Function to display query analysis
    function displayQueryAnalysis(data) {
        const resultsContainer = document.getElementById('recommendation-results');
        if (!resultsContainer || !data.analysis) return;

        // Clear previous analysis if exists
        const existingAnalysis = document.getElementById('query-analysis-container');
        if (existingAnalysis) {
            existingAnalysis.remove();
        }

        // Create and append analysis section
        const analysisDiv = document.createElement('div');
        analysisDiv.id = 'query-analysis-container';
        analysisDiv.className = 'query-analysis mb-4 p-3 bg-light rounded';
        analysisDiv.innerHTML = `
            <h3 class="h5 mb-3">
                <i class="fas fa-search me-2 text-primary"></i>
                Query Analysis
            </h3>
            <pre class="mb-0 p-3 bg-white rounded">${data.analysis}</pre>
        `;
        
        // Insert at the top of results container
        resultsContainer.insertBefore(analysisDiv, resultsContainer.firstChild);
    }

    // Function to display recommendations
    function displayRecommendations(data) {
        const resultsContainer = document.getElementById('recommendation-results');
        if (!resultsContainer) return;

        // Clear previous results (but keep analysis)
        const existingResults = document.querySelectorAll('#recommendation-results > *:not(#query-analysis-container)');
        existingResults.forEach(el => el.remove());

        // Create and append new results
        if (data.attractions && data.attractions.length > 0) {
            const attractionsSection = document.createElement('div');
            attractionsSection.className = 'mb-4';
            attractionsSection.innerHTML = `
                <h3 class="mb-3">
                    <i class="fas fa-landmark me-2 text-primary"></i>
                    Recommended Attractions in ${data.query}
                </h3>
                <div class="row">
                    ${data.attractions.map(attraction => `
                        <div class="col-md-4 mb-3">
                            <div class="card h-100">
                                <div class="card-body">
                                    <h5 class="card-title">${attraction.name || 'Unnamed Attraction'}</h5>
                                    <p class="text-warning">${attraction.rating_stars || ''}</p>
                                    <p class="card-text"><small class="text-muted">${attraction.city}, ${attraction.country}</small></p>
                                    <div class="card-text description">${attraction.description || 'No description available'}</div>
                                    ${attraction.webUrl ? `<a href="${attraction.webUrl}" target="_blank" class="btn btn-sm btn-outline-primary mt-2">Visit Website</a>` : ''}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
            resultsContainer.appendChild(attractionsSection);
        }

        if (data.hotels && data.hotels.length > 0) {
            const hotelsSection = document.createElement('div');
            hotelsSection.className = 'mb-4';
            hotelsSection.innerHTML = `
                <h3 class="mb-3">
                    <i class="fas fa-hotel me-2 text-success"></i>
                    Recommended Hotels in ${data.query}
                </h3>
                <div class="row">
                    ${data.hotels.map(hotel => `
                        <div class="col-md-4 mb-3">
                            <div class="card h-100">
                                <div class="card-body">
                                    <h5 class="card-title">${hotel.name || 'Unnamed Hotel'}</h5>
                                    <p class="text-warning">${hotel.rating_stars || ''}</p>
                                    <p class="card-text"><small class="text-muted">${hotel.city}, ${hotel.country}</small></p>
                                    <p class="card-text"><small>Amenities: ${hotel.amenities_formatted || 'None listed'}</small></p>
                                    <div class="card-text description">${hotel.description || 'No description available'}</div>
                                    ${hotel.webUrl ? `<a href="${hotel.webUrl}" target="_blank" class="btn btn-sm btn-outline-success mt-2">Visit Website</a>` : ''}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
            resultsContainer.appendChild(hotelsSection);
        }

        if ((!data.attractions || data.attractions.length === 0) && (!data.hotels || data.hotels.length === 0)) {
            const noResultsDiv = document.createElement('div');
            noResultsDiv.className = 'alert alert-info';
            noResultsDiv.innerHTML = `
                <i class="fas fa-info-circle me-2"></i>
                No recommendations found for "${data.query}". Try a different search.
            `;
            resultsContainer.appendChild(noResultsDiv);
        }

        // Reinitialize dynamic elements
        initializeDynamicElements();
    }

    // Show error message
    function showError(message) {
        const errorContainer = document.getElementById('error-container');
        const errorMessage = document.getElementById('error-message');
        
        if (errorContainer && errorMessage) {
            errorMessage.textContent = message;
            errorContainer.style.display = 'block';
        } else {
            alert(message); // Fallback if error container doesn't exist
        }
    }

    // Hide error message
    function hideError() {
        const errorContainer = document.getElementById('error-container');
        if (errorContainer) {
            errorContainer.style.display = 'none';
        }
    }

    // Initialize dynamic elements (tooltips, read more/less)
    function initializeDynamicElements() {
        // Tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });

        // Read more/less functionality
        document.querySelectorAll('.description').forEach(desc => {
            const fullText = desc.textContent;
            if (fullText.length > 200) {
                const truncated = fullText.substring(0, 200) + '...';
                desc.innerHTML = truncated + '<a href="#" class="read-more ms-1">Read more</a>';
                
                desc.querySelector('.read-more').addEventListener('click', function(e) {
                    e.preventDefault();
                    desc.innerHTML = fullText + '<a href="#" class="read-less ms-1">Read less</a>';
                    
                    desc.querySelector('.read-less').addEventListener('click', function(e) {
                        e.preventDefault();
                        desc.innerHTML = truncated + '<a href="#" class="read-more ms-1">Read more</a>';
                        initializeDynamicElements(); // Reattach event listeners
                    });
                });
            }
        });
    }

    // Country dropdown functionality
    const countryDropdown = document.getElementById('country');
    const cityDropdown = document.getElementById('city');
    
    if (countryDropdown && cityDropdown) {
        countryDropdown.addEventListener('change', function() {
            const selectedCountry = this.value;
            
            // Reset city dropdown
            cityDropdown.innerHTML = '<option value="">-- Select City --</option>';
            
            if (selectedCountry) {
                // Show loading state for cities
                const originalCityDropdown = cityDropdown.innerHTML;
                cityDropdown.disabled = true;
                cityDropdown.innerHTML = '<option value="">Loading cities...</option>';
                
                fetch(`/api/cities?country=${encodeURIComponent(selectedCountry)}`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Failed to load cities');
                        }
                        return response.json();
                    })
                    .then(data => {
                        cityDropdown.innerHTML = '<option value="">-- Select City --</option>';
                        data.cities.forEach(city => {
                            const option = document.createElement('option');
                            option.value = city;
                            option.textContent = city;
                            cityDropdown.appendChild(option);
                        });
                    })
                    .catch(error => {
                        console.error('Error fetching cities:', error);
                        cityDropdown.innerHTML = originalCityDropdown;
                        showError('Failed to load cities. Please try again.');
                    })
                    .finally(() => {
                        cityDropdown.disabled = false;
                    });
            }
        });
    }
});

// Add styles for the query analysis
const style = document.createElement('style');
style.textContent = `
    .query-analysis {
        background-color: #f8f9fa;
        border-left: 4px solid #0d6efd;
        border-radius: 0.25rem;
    }
    
    .query-analysis pre {
        white-space: pre-wrap;
        word-wrap: break-word;
        font-family: inherit;
        margin-bottom: 0;
    }
    
    .read-more, .read-less {
        color: #0d6efd;
        text-decoration: none;
        font-size: 0.9em;
    }
    
    .read-more:hover, .read-less:hover {
        text-decoration: underline;
    }
`;
document.head.appendChild(style);