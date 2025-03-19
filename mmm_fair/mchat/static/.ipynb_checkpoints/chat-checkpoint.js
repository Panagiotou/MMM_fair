document.addEventListener("DOMContentLoaded", function(){
    const chatBox = document.getElementById("chat-box");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const resetBtn = document.getElementById("reset-btn");
    const plot3dBox = document.getElementById("plot3d-box");
    const plot2dBox = document.getElementById("plot2d-box");
    const thetaInput = document.getElementById("theta-input");
    const thetaBtn = document.getElementById("theta-btn");
    const saveBtn = document.getElementById("save-btn");

    // Function to render messages in the chatbox
    function renderMessage(sender, text, isTyping = false){
        let messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;  // Auto-scroll

        if (isTyping) {
            typeText(messageDiv, text);
        } else {
            messageDiv.innerText = text;
        }
    }

    // Typing effect function
    function typeText(element, text, speed = 15) {
        let index = 0;
        function type() {
            if (index < text.length) {
                element.innerHTML += text.charAt(index);
                index++;
                setTimeout(type, speed);
            }
        }
        type();
    }

    // Show a welcome message with typing animation
    function showWelcomeMessage() {
        renderMessage("bot", "👋 Hello! Welcome to MMM-Fair Chat. \n At the moment, I can help you train either\n - 'MMM_Fair' (the AdaBoost style), or\n - 'MMM_Fair_GBT' (the Gradient Boosting style).\n\n Please type 'MMM_Fair' or 'MMM_Fair_GBT' to begin. \n Or type 'default' to run the GBT version on Adult data with default parameters.🙂 ", true);
    }

    // Reset chat history on page load
    function resetChatOnLoad() {
        fetch("/reset_chat")
        .then(() => {
            chatBox.innerHTML = "";
            plot3dBox.innerHTML = "";
            plot2dBox.innerHTML = "";
            showWelcomeMessage();
        })
        .catch(error => console.error("Error resetting chat:", error));
    }

    // Send message function
    function sendMessage(){
        let userMessage = chatInput.value.trim();
        if (!userMessage) return;
        chatInput.value = "";

        renderMessage("user", userMessage);

        fetch("/ask_chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message: userMessage})
        })
        .then(response => response.json())
        .then(data => {
            data.chat_history.forEach(chat => {
                renderMessage(chat.sender, chat.text, true);
            });

            // **Load Plot HTML Files Dynamically**
            if (data.plot_all_url) {
                console.log("DEBUG: Loading Plot3D from", data.plot_all_url);
                plot3dBox.innerHTML = `<iframe src="${data.plot_all_url}" width="100%" height="400px" frameborder="0"></iframe>`;
            }
            if (data.plot_fair_url) {
                console.log("DEBUG: Loading Plot2D from", data.plot_fair_url);
                plot2dBox.innerHTML = `<iframe src="${data.plot_fair_url}" width="100%" height="400px" frameborder="0"></iframe>`;
            }
        })
        .catch(error => console.error("Error:", error));
    }

    // Send message on button click
    sendBtn.addEventListener("click", sendMessage);

    // Send message on Enter key
    chatInput.addEventListener("keydown", function(e){
        if (e.key === "Enter") sendMessage();
    });

    // Reset Chat on button click
    resetBtn.addEventListener("click", function(){
        fetch("/reset_chat")
        .then(() => {
            chatBox.innerHTML = "";
            plot3dBox.innerHTML = "";
            plot2dBox.innerHTML = "";
            showWelcomeMessage();
        })
        .catch(error => console.error("Error resetting chat:", error));
    });
    // Saving Model as onnx for deployment
    
    
    // Handle Theta Selection and Model Update
    thetaBtn.addEventListener("click", function(){
        let thetaValue = thetaInput.value.trim();
        if (!thetaValue) {
            alert("Please enter a valid Theta index!");
            return;
        }

        fetch("/update_model", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({theta: thetaValue})
        })
        .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert(`Model updated successfully with Theta ${thetaValue}`);
        } else {
            alert(`Error updating model: ${data.error}`);
        }
    })
    .catch(error => console.error("Error updating model:", error));
    });  // <-- FIXED: Properly closed the function

    // Handle Save Model Button
    saveBtn.addEventListener("click", function(){
        let savePath = document.getElementById("save-path").value.trim();
        
        if (!savePath) {
            alert("Please enter a valid directory path to save the model!");
            return;
        }
    
        fetch("/save_model", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({save_path: savePath})
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert(`Model saved successfully in: ${savePath}`);
            } else {
                alert(`Error saving model: ${data.error}`);
            }
        })
        .catch(error => console.error("Error saving model:", error));
    });

    // Reset chat every time the page loads
    resetChatOnLoad();
});