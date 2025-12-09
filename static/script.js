document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault(); // Stop the default form submission

    const resultsDiv = document.getElementById('results');
    const blockchainDiv = document.getElementById('blockchain-activity');
    const fileInput = document.getElementById('binaryFile');

    if (fileInput.files.length === 0) {
        resultsDiv.innerHTML = '<p class="error">Please select a file first.</p>';
        return;
    }

    // 1. Prepare for submission and show loading
    resultsDiv.innerHTML = '<p class="loading">Analyzing file and updating blockchain... Please wait.</p>';
    blockchainDiv.innerHTML = '<p class="loading">Fetching recent blocks...</p>';
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
        // 2. Send the file to the Flask API
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        // 3. Handle errors from the server
        if (!response.ok) {
            resultsDiv.innerHTML = `<p class="error">**Analysis Error:** ${data.error || 'Server error occurred.'}</p>`;
            blockchainDiv.innerHTML = '';
            return;
        }

        // 4. Display the prediction results
        displayPrediction(data.prediction, data.overall_metrics, resultsDiv);

        // 5. Display the blockchain update
        displayBlockchain(data.blockchain, blockchainDiv);

    } catch (error) {
        // 6. Handle network or fetch errors
        console.error('Fetch error:', error);
        resultsDiv.innerHTML = `<p class="error">A network or fetch error occurred: ${error.message}</p>`;
        blockchainDiv.innerHTML = '';
    }
});

function displayPrediction(prediction, overallMetrics, targetElement) {
    let html = '<h3>Model Prediction</h3>';
    
    // Core Prediction
    html += '<div class="result-group">';
    html += `<strong>File:</strong> ${prediction.file || 'N/A'}<br>`;
    html += `<strong>Algorithm Name:</strong> ${prediction.algo_name || 'N/A'}<br>`;
    html += `<strong>Architecture Name:</strong> ${prediction.arch_name || 'N/A'}<br>`;
    html += `<strong>Protocol:</strong> ${prediction.protocol_name || 'N/A'}<br>`;
    html += `<strong>Is Proprietary (OCSVM):</strong> ${prediction.ocsvm_is_proprietary ? 'Yes' : 'No'}<br>`;
    html += `<strong>Prediction Score:</strong> ${prediction.ocsvm_score !== undefined ? prediction.ocsvm_score.toFixed(4) : 'N/A'}<br>`;
    html += '</div>';

    // Overall Model Metrics
    if (overallMetrics) {
        html += '<h3>Overall Model Metrics (Ensemble)</h3>';
        html += '<div class="result-group">';
        html += `<strong>Accuracy:</strong> ${overallMetrics.accuracy !== undefined ? overallMetrics.accuracy.toFixed(4) : 'N/A'}<br>`;
        html += `<strong>F1-Score:</strong> ${overallMetrics.f1_score !== undefined ? overallMetrics.f1_score.toFixed(4) : 'N/A'}<br>`;
        html += `<strong>Precision:</strong> ${overallMetrics.precision !== undefined ? overallMetrics.precision.toFixed(4) : 'N/A'}<br>`;
        html += `<strong>Recall:</strong> ${overallMetrics.recall !== undefined ? overallMetrics.recall.toFixed(4) : 'N/A'}<br>`;
        html += '</div>';
    }

    // Raw Features (for inspection)
    if (prediction.input_features) {
        html += '<h4>Input Features (Excerpt)</h4>';
        // Display features in a pre-formatted block for easy reading
        html += `<pre>${JSON.stringify(prediction.input_features, null, 2).substring(0, 500)}...</pre>`;
    }

    targetElement.innerHTML = html;
}

function displayBlockchain(blockchain, targetElement) {
    let html = '<h3>New Block Added</h3>';
    html += renderBlock(blockchain.new_block, true); // Highlight the new block

    html += '<h3>Last 4 Blocks in Chain</h3>';
    if (blockchain.last_blocks.length > 1) {
        // Skip the very last block, as it's the one we just added (the new_block)
        const recent = blockchain.last_blocks.slice(0, -1); 
        recent.reverse().forEach(block => {
            html += renderBlock(block, false);
        });
    } else {
        html += '<p>Only the new block is currently available.</p>';
    }

    targetElement.innerHTML = html;
}

function renderBlock(block, isNew) {
    const style = isNew ? 'border: 2px solid green; background-color: #e6ffe6;' : 'border: 1px solid #ddd;';
    let html = `<div style="${style} padding: 10px; margin-bottom: 10px; border-radius: 4px;">`;
    html += `<strong>Index:</strong> ${block.index}<br>`;
    html += `<strong>Timestamp:</strong> ${new Date(block.timestamp * 1000).toLocaleString()}<br>`;
    html += `<strong>File:</strong> ${block.data.file || 'N/A'}<br>`;
    html += `<strong>Protocol:</strong> ${block.data.protocol_name || 'N/A'}<br>`;
    html += `<strong>Hash:</strong> <pre style="font-size: small; margin-top: 5px; margin-bottom: 0;">${block.hash}</pre>`;
    html += `<strong>Previous Hash:</strong> <pre style="font-size: small; margin-top: 5px; margin-bottom: 0;">${block.previous_hash}</pre>`;
    html += '</div>';
    return html;
}