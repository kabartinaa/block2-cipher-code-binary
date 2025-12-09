document.getElementById("uploadBtn").addEventListener("click", () => {
    const fileInput = document.getElementById("fileInput");
    const status = document.getElementById("uploadStatus");

    if (!fileInput.files.length) {
        status.textContent = "Please select a file first.";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    status.textContent = "Uploading and analyzing...";

    fetch("/api/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            status.textContent = "Backend error: " + data.error;
            console.error("Backend error:", data.error);
            return;
        }

        status.textContent = "Analysis completed.";
        renderDashboard(data);
    })
    .catch(err => {
        status.textContent = "Network error.";
        console.error(err);
    });
});


// -------------------------------
// SAFE RENDER FUNCTION
// -------------------------------
function renderDashboard(res) {

    // Helper: safely get values without crashing
    const get = (obj, key, fallback = "-") => obj && obj[key] !== undefined ? obj[key] : fallback;

    document.getElementById("fileName").textContent = res.file || "Uploaded file";

    document.getElementById("algoName").textContent = get(res, "algo_name");
    document.getElementById("algoId").textContent = `(id = ${get(res, "algo_id")})`;
    document.getElementById("archName").textContent = get(res, "arch_name");
    document.getElementById("protoName").textContent = get(res, "protocol_name");

    // proprietary badge
    const badge = document.getElementById("proprietaryBadge");
    if (res.ocsvm_is_proprietary) {
        badge.textContent = "Proprietary / Outlier";
        badge.classList.remove("badge-success");
        badge.classList.add("badge-danger");
    } else {
        badge.textContent = "Standard-like";
        badge.classList.remove("badge-danger");
        badge.classList.add("badge-success");
    }

    document.getElementById("algoBadge").textContent = get(res, "algo_name");

    document.getElementById("gnnVote").textContent = get(res, "gnn_pred_algo_id");
    document.getElementById("lstmVote").textContent = get(res, "lstm_pred_algo_id");
    document.getElementById("xgbVote").textContent = get(res, "xgb_algo_pred_algo_id");

    // PROTOCOL LIST
    const steps = document.getElementById("protoSteps");
    steps.innerHTML = "";
    (res.protocol_sequence || []).forEach(step => {
        const li = document.createElement("li");
        li.textContent = step;
        steps.appendChild(li);
    });

    // FEATURE TABLE
    const featureOrder = ["length", "unique_bytes", "entropy", "mean_byte", "std_byte"];
    const featureTable = document.getElementById("featureTableBody");
    featureTable.innerHTML = "";

    featureOrder.forEach((name, i) => {
        const tr = document.createElement("tr");

        tr.innerHTML = `
            <td>${name}</td>
            <td>${res.input_features ? get(res.input_features, name) : "-"}</td>
            <td>${res.algo_mean_features ? res.algo_mean_features[i] : "-"}</td>
        `;

        featureTable.appendChild(tr);
    });
}
