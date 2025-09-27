// // script.js - JavaScript for SpeakWise Application

// // Navigation Functions
// function navigateToLogin() {
//     window.location.href = 'login.html';
// }

// function goBack() {
//     window.location.href = 'index.html';
// }

// // Homepage functionality
// document.addEventListener('DOMContentLoaded', function() {
//     // Check if we're on the homepage
//     const getStartedBtn = document.querySelector('.get-started-btn');
//     if (getStartedBtn) {
//         getStartedBtn.addEventListener('click', function(e) {
//             e.preventDefault();
//             navigateToLogin();
//         });
//     }
    
//     // Login form functionality
//     const loginForm = document.getElementById('loginForm');
//     if (loginForm) {
//         loginForm.addEventListener('submit', function(e) {
//             e.preventDefault();
//             handleLogin();
//         });
//     }
    
//     // Social login buttons
//     const googleBtn = document.querySelector('.google-btn');
//     const facebookBtn = document.querySelector('.facebook-btn');
    
//     if (googleBtn) {
//         googleBtn.addEventListener('click', function() {
//             handleSocialLogin('google');
//         });
//     }
    
//     if (facebookBtn) {
//         facebookBtn.addEventListener('click', function() {
//             handleSocialLogin('facebook');
//         });
//     }
    
//     // Forgot password link
//     const forgotPasswordLink = document.querySelector('.forgot-password');
//     if (forgotPasswordLink) {
//         forgotPasswordLink.addEventListener('click', function(e) {
//             e.preventDefault();
//             handleForgotPassword();
//         });
//     }
    
//     // Sign up link
//     const signupLink = document.querySelector('.signup-link .link');
//     if (signupLink) {
//         signupLink.addEventListener('click', function(e) {
//             e.preventDefault();
//             handleSignup();
//         });
//     }
// });

// // Login form handler
// function handleLogin() {
//     const email = document.getElementById('email').value;
//     const password = document.getElementById('password').value;
//     const remember = document.getElementById('remember').checked;
    
//     // Basic validation
//     if (!email || !password) {
//         showMessage('Please fill in all fields', 'error');
//         return;
//     }
    
//     if (!isValidEmail(email)) {
//         showMessage('Please enter a valid email address', 'error');
//         return;
//     }
    
//     // Show loading state
//     const loginBtn = document.querySelector('.login-btn');
//     const originalText = loginBtn.textContent;
//     loginBtn.textContent = 'Signing In...';
//     loginBtn.disabled = true;
    
//     // Simulate API call (replace with actual authentication)
//     setTimeout(() => {
//         // Mock successful login
//         if (email === 'demo@speakwise.com' && password === 'demo123') {
//             showMessage('Login successful! Redirecting...', 'success');
//             setTimeout(() => {
//                 // Redirect to dashboard (you can create this later)
//                 window.location.href = 'dashboard.html';
//             }, 1500);
//         } else {
//             showMessage('Invalid email or password', 'error');
//             loginBtn.textContent = originalText;
//             loginBtn.disabled = false;
//         }
//     }, 2000);
// }

// // Social login handler
// function handleSocialLogin(provider) {
//     showMessage(`Connecting to ${provider}...`, 'info');
    
//     // Simulate social login process
//     setTimeout(() => {
//         showMessage(`${provider} login successful! Redirecting...`, 'success');
//         setTimeout(() => {
//             window.location.href = 'dashboard.html';
//         }, 1500);
//     }, 2000);
// }

// // Forgot password handler
// function handleForgotPassword() {
//     const email = prompt('Enter your email address to reset password:');
//     if (email && isValidEmail(email)) {
//         showMessage('Password reset link sent to your email!', 'success');
//     } else if (email) {
//         showMessage('Please enter a valid email address', 'error');
//     }
// }

// // Sign up handler
// function handleSignup() {
//     showMessage('Redirecting to sign up...', 'info');
//     // You can create a signup.html page later
//     setTimeout(() => {
//         alert('Sign up page will be created. For now, use demo@speakwise.com / demo123 to login.');
//     }, 1000);
// }

// // Utility functions
// function isValidEmail(email) {
//     const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
//     return emailRegex.test(email);
// }

// function showMessage(message, type = 'info') {
//     // Remove existing messages
//     const existingMessage = document.querySelector('.message');
//     if (existingMessage) {
//         existingMessage.remove();
//     }
    
//     // Create message element
//     const messageDiv = document.createElement('div');
//     messageDiv.className = `message ${type}`;
//     messageDiv.textContent = message;
    
//     // Style the message
//     messageDiv.style.cssText = `
//         position: fixed;
//         top: 20px;
//         right: 20px;
//         padding: 15px 20px;
//         border-radius: 8px;
//         color: white;
//         font-weight: 500;
//         z-index: 1000;
//         transform: translateX(100%);
//         transition: transform 0.3s ease;
//         max-width: 300px;
//         word-wrap: break-word;
//     `;
    
//     // Set background color based on type
//     switch(type) {
//         case 'success':
//             messageDiv.style.background = 'linear-gradient(145deg, #4a7474, #77b3b3)';
//             break;
//         case 'error':
//             messageDiv.style.background = 'linear-gradient(145deg, #d32f2f, #f44336)';
//             break;
//         case 'info':
//         default:
//             messageDiv.style.background = 'linear-gradient(145deg, #395f5f, #4a7474)';
//             break;
//     }
    
//     // Add to page
//     document.body.appendChild(messageDiv);
    
//     // Animate in
//     setTimeout(() => {
//         messageDiv.style.transform = 'translateX(0)';
//     }, 100);
    
//     // Remove after 4 seconds
//     setTimeout(() => {
//         messageDiv.style.transform = 'translateX(100%)';
//         setTimeout(() => {
//             if (messageDiv.parentNode) {
//                 messageDiv.parentNode.removeChild(messageDiv);
//             }
//         }, 300);
//     }, 4000);
// }

// // Add some interactive effects
// document.addEventListener('DOMContentLoaded', function() {
//     // Add floating animation to background circles (if on homepage)
//     if (document.querySelector('.get-started-btn')) {
//         addFloatingAnimation();
//     }
    
//     // Add input focus effects
//     const inputs = document.querySelectorAll('.form-input');
//     inputs.forEach(input => {
//         input.addEventListener('focus', function() {
//             this.parentNode.style.transform = 'scale(1.02)';
//         });
        
//         input.addEventListener('blur', function() {
//             this.parentNode.style.transform = 'scale(1)';
//         });
//     });
// });

// function addFloatingAnimation() {
//     // Add subtle floating animation to content only (not background)
//     const content = document.querySelector('.content');
//     if (content) {
//         let mouseX = 0;
//         let mouseY = 0;
        
//         document.addEventListener('mousemove', function(e) {
//             mouseX = (e.clientX / window.innerWidth) * 2 - 1;
//             mouseY = (e.clientY / window.innerHeight) * 2 - 1;
            
//             const moveX = mouseX * 3; // Reduced movement
//             const moveY = mouseY * 3; // Reduced movement
            
//             content.style.transform = `translate(${moveX}px, ${moveY}px)`;
//         });
//     }
// }

// // Console welcome message
// console.log(`
// 🎤 Welcome to SpeakWise! 
// 📧 Demo login: demo@speakwise.com
// 🔑 Demo password: demo123
// `);




// script.js - JavaScript for SpeakWise Application

// --- GLOBAL VARIABLES for Dashboard ---
let uploadInterval; // Manages the simulated upload process
let cameraStream;   // Manages the camera stream for OpenCV modal

// --- NAVIGATION & HOMEPAGE ---

// Navigation Functions
function navigateToLogin() {
    window.location.href = 'login.html';
}

function goBack() {
    window.location.href = 'index.html';
}

// Homepage and Login Form Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Homepage: Get Started button
    const getStartedBtn = document.querySelector('.get-started-btn');
    if (getStartedBtn) {
        getStartedBtn.addEventListener('click', (e) => {
            e.preventDefault();
            navigateToLogin();
        });
    }
    
    // Login Page: Form submission
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            handleLogin();
        });
    }
    
    // Login Page: Social login buttons
    const googleBtn = document.querySelector('.google-btn');
    if (googleBtn) {
        googleBtn.addEventListener('click', () => handleSocialLogin('google'));
    }
    
    const facebookBtn = document.querySelector('.facebook-btn');
    if (facebookBtn) {
        facebookBtn.addEventListener('click', () => handleSocialLogin('facebook'));
    }
    
    // Login Page: Links
    const forgotPasswordLink = document.querySelector('.forgot-password');
    if (forgotPasswordLink) {
        forgotPasswordLink.addEventListener('click', (e) => {
            e.preventDefault();
            handleForgotPassword();
        });
    }
    
    const signupLink = document.querySelector('.signup-link .link');
    if (signupLink) {
        signupLink.addEventListener('click', (e) => {
            e.preventDefault();
            handleSignup();
        });
    }

    // Login Page: Input focus effects
    const inputs = document.querySelectorAll('.form-input');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentNode.style.transition = 'transform 0.2s ease';
            this.parentNode.style.transform = 'scale(1.02)';
        });
        
        input.addEventListener('blur', function() {
            this.parentNode.style.transform = 'scale(1)';
        });
    });
});

// --- AUTHENTICATION HANDLERS ---

// Login form handler
function handleLogin() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    // Basic validation
    if (!email || !password) {
        showMessage('Please fill in all fields.', 'error');
        return;
    }
    if (!isValidEmail(email)) {
        showMessage('Please enter a valid email address.', 'error');
        return;
    }
    
    // Show loading state on button
    const loginBtn = document.querySelector('.login-btn');
    const originalText = loginBtn.textContent;
    loginBtn.textContent = 'Signing In...';
    loginBtn.disabled = true;
    
    // Simulate API call for authentication
    setTimeout(() => {
        if (email === 'demo@speakwise.com' && password === 'demo123') {
            showMessage('Login successful! Redirecting...', 'success');
            setTimeout(() => {
                window.location.href = 'dashboard.html';
            }, 1500);
        } else {
            showMessage('Invalid email or password.', 'error');
            loginBtn.textContent = originalText;
            loginBtn.disabled = false;
        }
    }, 2000);
}

// Social login handler
function handleSocialLogin(provider) {
    showMessage(`Connecting to ${provider}...`, 'info');
    setTimeout(() => {
        showMessage(`${provider} login successful! Redirecting...`, 'success');
        setTimeout(() => {
            window.location.href = 'dashboard.html';
        }, 1500);
    }, 2000);
}

// Forgot password handler
function handleForgotPassword() {
    const email = prompt('Enter your email address to reset your password:');
    if (email === null) return; // User cancelled prompt
    if (email && isValidEmail(email)) {
        showMessage('Password reset link sent to your email!', 'success');
    } else {
        showMessage('Please enter a valid email address.', 'error');
    }
}

// Sign up handler
function handleSignup() {
    showMessage('Redirecting to sign up page...', 'info');
    setTimeout(() => {
        alert('The Sign Up page is not yet created. For now, please use the demo credentials to log in.');
    }, 1000);
}


// --- DASHBOARD FUNCTIONALITY ---

// Header actions
function logout() {
    showMessage('Logging you out...', 'info');
    setTimeout(() => {
        window.location.href = 'index.html';
    }, 1500);
}

function showProfile() {
    alert('The Profile page or modal would be displayed here.');
}

// File upload triggers
function triggerUpload(type) {
    document.getElementById(`${type}Input`).click();
}

// Handles file selection and simulates upload progress
function handleFileUpload(event, type) {
    const file = event.target.files[0];
    if (!file) return;

    const modal = document.getElementById('uploadModal');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    modal.style.display = 'flex';
    progressFill.style.width = '0%';
    progressText.textContent = '0% Complete';

    // Simulate upload progress
    let progress = 0;
    uploadInterval = setInterval(() => {
        progress += 10;
        if (progress > 100) progress = 100;

        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${progress}% Complete`;

        if (progress >= 100) {
            clearInterval(uploadInterval);
            setTimeout(() => {
                modal.style.display = 'none';
                showMessage(`"${file.name}" uploaded successfully!`, 'success');
            }, 500);
        }
    }, 250);
}

function cancelUpload() {
    clearInterval(uploadInterval);
    document.getElementById('uploadModal').style.display = 'none';
    showMessage('Upload cancelled.', 'info');
}

// OpenCV analysis modal with camera access
async function openCVAnalysis() {
    const modal = document.getElementById('opencvModal');
    const videoElement = document.getElementById('cameraFeed');
    if (!modal || !videoElement) return;

    modal.style.display = 'flex';
    
    try {
        // Request access to the user's camera
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            videoElement.srcObject = cameraStream;
            showMessage('Camera activated for analysis.', 'info');
        }
    } catch (err) {
        console.error("Error accessing camera:", err);
        showMessage('Could not access camera. Please check permissions.', 'error');
        closeOpenCV(); // Close modal if camera fails
    }
}

function closeOpenCV() {
    const modal = document.getElementById('opencvModal');
    
    // Stop all camera stream tracks to turn off the camera light
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }
    
    if (modal) {
        modal.style.display = 'none';
    }
}

// Placeholder functions for OpenCV and Quick Actions
function startAnalysis() { alert('Real-time analysis would start now, processing the video feed.'); }
function stopAnalysis() { alert('Analysis stopped. You can now save the results.'); }
function saveAnalysis() { alert('Analysis results would be saved.'); }
function startRecording() { alert('Real-time voice recording would start here.'); }
function practiceMode() { alert('Entering Practice Mode.'); }
function viewReports() { alert('Redirecting to the reports and analytics page.'); }
function settings() { alert('Opening the settings panel.'); }


// --- UTILITY & UI FUNCTIONS ---

// Email validation
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// On-screen message display
function showMessage(message, type = 'info') {
    const existingMessage = document.querySelector('.message');
    if (existingMessage) existingMessage.remove();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = message;
    
    // Apply styles via JS for portability
    messageDiv.style.cssText = `
        position: fixed; top: 20px; right: 20px; padding: 15px 20px; border-radius: 8px;
        color: white; font-weight: 500; z-index: 2000; max-width: 320px; word-wrap: break-word;
        transform: translateX(120%); transition: transform 0.4s ease-in-out;`;
    
    const colors = {
        success: 'linear-gradient(145deg, #4a7474, #77b3b3)',
        error: 'linear-gradient(145deg, #d32f2f, #f44336)',
        info: 'linear-gradient(145deg, #395f5f, #4a7474)'
    };
    messageDiv.style.background = colors[type] || colors.info;
    
    document.body.appendChild(messageDiv);
    
    setTimeout(() => { messageDiv.style.transform = 'translateX(0)'; }, 100);
    
    setTimeout(() => {
        messageDiv.style.transform = 'translateX(120%)';
        setTimeout(() => { messageDiv.remove(); }, 400);
    }, 4000);
}
// script.js

// --- SLIDER FUNCTIONALITY ---
document.addEventListener('DOMContentLoaded', () => {
    const sliderContainer = document.getElementById('upload-slider');
    if (!sliderContainer) return;

    const wrapper = sliderContainer.querySelector('.slider-wrapper');
    const nextBtn = sliderContainer.querySelector('.next-btn');
    const prevBtn = sliderContainer.querySelector('.prev-btn');
    const slides = sliderContainer.querySelectorAll('.upload-card');
    
    let currentIndex = 0;
    let slidesVisible = 1;

    function updateSlidesVisible() {
        // Calculate how many slides are currently visible based on container and slide width
        const containerWidth = sliderContainer.offsetWidth;
        if (slides.length === 0) return;
        const slideWidth = slides[0].offsetWidth;
        const gap = parseInt(window.getComputedStyle(wrapper).gap) || 0;
        
        // Ensure we don't divide by zero
        const totalSlideWidth = slideWidth + gap;
        if (totalSlideWidth > 0) {
            slidesVisible = Math.floor(containerWidth / totalSlideWidth);
        }
        if (slidesVisible < 1) slidesVisible = 1; // Always at least 1 is visible
    }

    function updateSlider() {
        if (slides.length === 0) return;
        const slideWidth = slides[0].offsetWidth;
        const gap = parseInt(window.getComputedStyle(wrapper).gap) || 0;
        const offset = -currentIndex * (slideWidth + gap);
        wrapper.style.transform = `translateX(${offset}px)`;

        // Enable or disable buttons based on position
        prevBtn.disabled = currentIndex === 0;
        nextBtn.disabled = currentIndex >= slides.length - slidesVisible;
    }

    nextBtn.addEventListener('click', () => {
        // Move to the next slide
        if (currentIndex < slides.length - slidesVisible) {
            currentIndex++;
            updateSlider();
        }
    });

    prevBtn.addEventListener('click', () => {
        // Move to the previous slide
        if (currentIndex > 0) {
            currentIndex--;
            updateSlider();
        }
    });

    // Recalculate slider dimensions on window resize
    function initSlider() {
        updateSlidesVisible();
        // Adjust currentIndex if it's out of bounds after resize
        if (currentIndex > slides.length - slidesVisible) {
            currentIndex = slides.length - slidesVisible;
        }
        updateSlider();
    }
    
    window.addEventListener('resize', initSlider);
    
    // Use a small timeout to ensure all styles are loaded before calculating dimensions
    setTimeout(initSlider, 100);
});

// Console welcome message
console.log(`
🎤 Welcome to SpeakWise! 
📧 Demo login: demo@speakwise.com
🔑 Demo password: demo123
`);



