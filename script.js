function predict(){
    const socket = new WebSocket("ws://localhost:8765");
    socket.onopen = function(event) {
        console.log("WebSocket connection opened.");
        const inputValue = document.getElementById('textInput').value;
        socket.send(inputValue);
    };

    socket.onmessage = function(event) {
        console.log("Received message from Python:", event.data.toString());
        const result = event.data;
        const par = document.getElementById('prediction');
        par.style.display = 'block';
        par.innerHTML = result;
        document.getElementById('loader').style.display = 'none';
    };

    socket.onclose = function(event) {
        console.log("WebSocket connection closed.");
    };

    socket.onerror = function(error) {
        console.log("WebSocket error: ", error);
    };

    document.getElementById('loader').style.display = 'block';
}