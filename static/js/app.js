// Initialize AOS animation library
document.addEventListener('DOMContentLoaded', function() {
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: true,
        mirror: false
    });

    // Animate statistics numbers
    animateStats();
});

// Stats counter animation
function animateStats() {
    const statElements = document.querySelectorAll('.stat-value');
    
    statElements.forEach(stat => {
        const target = parseInt(stat.getAttribute('data-count'));
        const duration = 2000; // 2 seconds
        const step = target / 100;
        let current = 0;
        
        const updateCounter = () => {
            current += step;
            if (current < target) {
                stat.textContent = Math.floor(current);
                requestAnimationFrame(updateCounter);
            } else {
                stat.textContent = target;
            }
        };
        
        requestAnimationFrame(updateCounter);
    });
}

// Pywebview API functions
function startROI() {
    showLoading();
    pywebview.api.start_roi().then(response => {
        hideLoading();
        showNotification(response.message, 'success');
    }).catch(error => {
        hideLoading();
        showNotification('Error starting ROI definition', 'error');
    });
}

function startProcess() {
    showLoading();
    pywebview.api.start_process().then(response => {
        hideLoading();
        showNotification(response.message, 'success');
    }).catch(error => {
        hideLoading();
        showNotification('Error starting process definition', 'error');
    });
}

function startUserMode() {
    showLoading();
    pywebview.api.start_user_mode().then(response => {
        hideLoading();
        showNotification(response.message, 'success');
    }).catch(error => {
        hideLoading();
        showNotification('Error starting user mode', 'error');
    });
}

// Loading overlay
function showLoading() {
    // Create loading overlay if it doesn't exist
    if (!document.getElementById('loading-overlay')) {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner"></div>
                <p>Loading...</p>
            </div>
        `;
        document.body.appendChild(overlay);
        
        // Add required CSS
        const style = document.createElement('style');
        style.textContent = `
            #loading-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.7);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            .loading-spinner {
                text-align: center;
                color: white;
            }
            .spinner {
                width: 50px;
                height: 50px;
                border: 5px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top-color: #3498db;
                animation: spin 1s ease-in-out infinite;
                margin: 0 auto 15px;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    }
    
    // Show overlay with fade in
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = 'flex';
    setTimeout(() => {
        overlay.style.opacity = '1';
    }, 10);
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.opacity = '0';
        setTimeout(() => {
            overlay.style.display = 'none';
        }, 300);
    }
}

// Notification system
function showNotification(message, type = 'info') {
    // Create notification container if it doesn't exist
    if (!document.getElementById('notification-container')) {
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9998;
        `;
        document.body.appendChild(container);
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    // Set icon based on type
    let icon = 'info-circle';
    if (type === 'success') icon = 'check-circle';
    if (type === 'error') icon = 'exclamation-circle';
    if (type === 'warning') icon = 'exclamation-triangle';
    
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${icon}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add required CSS
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            .notification {
                background-color: white;
                color: #333;
                padding: 15px;
                border-radius: 6px;
                box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
                margin-bottom: 15px;
                width: 300px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transform: translateX(400px);
                opacity: 0;
                transition: all 0.5s ease;
            }
            .notification.show {
                transform: translateX(0);
                opacity: 1;
            }
            .notification-content {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .notification.info i { color: #3498db; }
            .notification.success i { color: #2ecc71; }
            .notification.error i { color: #e74c3c; }
            .notification.warning i { color: #f39c12; }
            .notification-close {
                background: none;
                border: none;
                cursor: pointer;
                font-size: 14px;
                padding: 5px;
                color: #777;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Add to container
    const container = document.getElementById('notification-container');
    container.appendChild(notification);
    
    // Show with animation
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // Setup close button
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 500);
    });
    
    // Auto-close after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 500);
        }
    }, 5000);
}

// Load pages dynamically
function loadPage(page) {
    showLoading();
    fetch(`templates/${page}.html`)
        .then(response => response.text())
        .then(html => {
            document.querySelector('main').innerHTML = html;
            hideLoading();
            
            // Re-initialize AOS for new content
            AOS.refresh();
            
            // If loading dashboard, initialize its specific functions
            if (page === 'dashboard') {
                loadDashboard();
            }
        })
        .catch(error => {
            hideLoading();
            showNotification('Error loading page', 'error');
        });
}

// Handle login
function handleLogin() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    // Show loading animation
    showLoading();
    
    // Simulate API call with timeout
    setTimeout(() => {
        hideLoading();
        
        if (username === 'admin' && password === 'password') {
            showNotification('Login Successful', 'success');
            loadPage('dashboard');
        } else {
            showNotification('Invalid Credentials', 'error');
            
            // Shake the login form
            const form = document.querySelector('.login-form');
            form.classList.add('shake-animation');
            setTimeout(() => {
                form.classList.remove('shake-animation');
            }, 500);
        }
    }, 1000);
    
    return false; // Prevent form submission
}

// Update dashboard stats
function updateStats(total, correct, incorrect) {
    // Get elements
    const totalElement = document.getElementById('total-cycles');
    const correctElement = document.getElementById('correct-cycles');
    const incorrectElement = document.getElementById('incorrect-cycles');
    
    if (totalElement && correctElement && incorrectElement) {
        // Animate the stats
        animateCounter(totalElement, total);
        animateCounter(correctElement, correct);
        animateCounter(incorrectElement, incorrect);
        
        // Update chart if it exists
        updateChart(total, correct, incorrect);
    }
}

function animateCounter(element, target) {
    let current = 0;
    const increment = Math.max(1, Math.floor(target / 50));
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = current;
        }
    }, 20);
}

function loadDashboard() {
    pywebview.api.get_stats().then(stats => {
        updateStats(stats.total, stats.correct, stats.incorrect);
    }).catch(error => {
        showNotification('Error loading dashboard data', 'error');
    });
}

// Function to toggle between dark and light theme
function toggleTheme() {
    const body = document.body;
    const toggleButton = document.getElementById("theme-toggle");
    const isDarkMode = body.classList.toggle("dark-theme");

    // Update button text and icon
    if (isDarkMode) {
        toggleButton.innerHTML = '<i class="fas fa-moon"></i> Dark Mode';
    } else {
        toggleButton.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
    }

    // Save the user's preference
    localStorage.setItem("theme", isDarkMode ? "dark" : "light");
}

// Apply the user's saved theme preference on page load
window.addEventListener("DOMContentLoaded", () => {
    const savedTheme = localStorage.getItem("theme") || "light";
    if (savedTheme === "dark") {
        document.body.classList.add("dark-theme");
        document.getElementById("theme-toggle").innerHTML = '<i class="fas fa-moon"></i> Dark Mode';
    }
    
    // Initialize page animations and effects
    initializeEffects();
});

// Initialize additional page effects
function initializeEffects() {
    // Add parallax effect to hero section
    document.addEventListener('mousemove', function(e) {
        const heroSection = document.querySelector('.hero-section');
        if (heroSection) {
            const moveX = (e.clientX - window.innerWidth / 2) * 0.01;
            const moveY = (e.clientY - window.innerHeight / 2) * 0.01;
            
            heroSection.style.backgroundPosition = `calc(50% + ${moveX}px) calc(50% + ${moveY}px)`;
        }
    });
    
    // Add hover effects to feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}