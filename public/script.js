// Global state
let bookSelected = false;
let isProcessing = false;
let selectedBookName = null; // Store the selected book name for serverless requests

// Use relative paths - works on same domain (Vercel serves both frontend and API)
const BASE_URL = "";

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Add event listener for Enter key in question input
    document.getElementById('question-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !isProcessing) {
            askQuestion();
        }
    });
});

// Function to handle book selection
function selectBook() {
    const bookSelect = document.getElementById('book-select');
    const bookName = bookSelect.value;

    if (!bookName) {
        showNotification('Please select a book first', 'error');
        return;
    }

    document.getElementById('processing-message').style.display = 'block';
    document.getElementById('book-selection').style.display = 'none';

    fetch(`${BASE_URL}/select_book`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ book_name: bookName }),
    })
    .then(handleResponse)
    .then(data => {
        bookSelected = true;
        selectedBookName = data.book_name || bookName; // Store book name from response
        checkStatus();
    })
    .catch(handleError);
}

function checkStatus() {
    fetch(`${BASE_URL}/status`)
    .then(handleResponse)
    .then(data => {
        if (data.status === 'ready') {
            document.getElementById('processing-message').style.display = 'none';
            document.getElementById('chat-interface').style.display = 'flex';
            addMessage("Hello! I'm ready to discuss the book with you. What would you like to know?", 'bot-message');
        } else {
            setTimeout(checkStatus, 1000);
        }
    })
    .catch(handleError);
}


// Function to handle user questions
function askQuestion() {
    if (isProcessing) return;

    const questionInput = document.getElementById('question-input');
    const question = questionInput.value.trim();

    if (!question) {
        showNotification('Please enter a question', 'error');
        return;
    }

    if (!selectedBookName) {
        showNotification('Please select a book first', 'error');
        return;
    }

    isProcessing = true;
    addMessage(question, 'user-message');
    questionInput.value = '';

    // Add typing indicator
    const typingId = addTypingIndicator();

    // Include book_name in request (required for serverless/stateless API)
    fetch(`${BASE_URL}/ask`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question,
            book_name: selectedBookName
        }),
    })
    .then(handleResponse)
    .then(data => {
        removeTypingIndicator(typingId);
        addMessage(data.answer, 'bot-message', true);
    })
    .catch(error => {
        removeTypingIndicator(typingId);
        handleError(error);
    })
    .finally(() => {
        isProcessing = false;
    });
}

// Function to add a message to the chat interface
function addMessage(text, className, isMarkdown = false) {
    if (!text) return;

    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;

    if (isMarkdown && typeof marked !== 'undefined') {
        messageDiv.innerHTML = marked.parse(text);
        // Add syntax highlighting if available
        if (window.Prism) {
            messageDiv.querySelectorAll('pre code').forEach((block) => {
                Prism.highlightElement(block);
            });
        }
    } else {
        messageDiv.textContent = text;
    }

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Function to add typing indicator
function addTypingIndicator() {
    const id = 'typing-' + Date.now();
    const indicator = document.createElement('div');
    indicator.id = id;
    indicator.className = 'message bot-message typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    document.getElementById('chat-messages').appendChild(indicator);
    scrollToBottom();
    return id;
}

// Function to remove typing indicator
function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

// Function to scroll chat to bottom
function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        // Use requestAnimationFrame for smooth scrolling
        requestAnimationFrame(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });
    }
}

// Function to show notifications
function showNotification(message, type = 'info') {
    // You can implement a toast notification system here
    alert(message);
}

// Function to handle fetch responses
function handleResponse(response) {
    if (!response.ok) {
        return response.json().then(err => Promise.reject(err));
    }
    return response.json();
}

// Function to handle errors
function handleError(error) {
    console.error('Error:', error);
    showNotification(error.error || error.message || 'An error occurred. Please try again.', 'error');
}

// Function to end session
function endSession() {
    if (confirm('Are you sure you want to end this session?')) {
        fetch(`${BASE_URL}/end_session`, {
            method: 'POST',
        })
        .then(handleResponse)
        .then(data => {
            showNotification(data.message, 'success');
            setTimeout(() => location.reload(), 1000);
        })
        .catch(handleError);
    }
}
