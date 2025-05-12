document.addEventListener('DOMContentLoaded', () => {
    const uploadButton = document.getElementById('uploadButton');
    const pdfFile = document.getElementById('pdfFile');
    const uploadStatus = document.getElementById('uploadStatus');

    const askButton = document.getElementById('askButton');
    const questionInput = document.getElementById('questionInput');
    const pdfSelect = document.getElementById('pdfSelect');
    const answerText = document.getElementById('answerText');
    const sourceDocumentsList = document.getElementById('sourceDocumentsList');

    const listPdfsButton = document.getElementById('listPdfsButton');
    const pdfsTableBody = document.getElementById('pdfsTableBody');

    const API_BASE_URL = 'http://localhost:8080'; // Assuming Go HTTP server runs on 8080

    // Function to fetch and list PDFs
    async function fetchAndListPDFs() {
        try {
            const response = await fetch(`${API_BASE_URL}/pdfs`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            pdfsTableBody.innerHTML = ''; // Clear existing rows
            pdfSelect.innerHTML = '<option value="">All Documents</option>'; // Reset select

            if (data.documents && data.documents.length > 0) {
                data.documents.forEach(doc => {
                    const row = pdfsTableBody.insertRow();
                    row.insertCell().textContent = doc.file_name || 'N/A';
                    row.insertCell().textContent = doc.document_id || 'N/A';
                    row.insertCell().textContent = doc.title || 'N/A';
                    row.insertCell().textContent = doc.author || 'N/A';
                    row.insertCell().textContent = doc.total_pages || 'N/A';
                    row.insertCell().textContent = doc.total_chunks || 'N/A';
                    row.insertCell().textContent = doc.processed_at ? new Date(doc.processed_at).toLocaleString() : 'N/A';
                    row.insertCell().textContent = doc.status || 'N/A';

                    // Populate select dropdown
                    const option = document.createElement('option');
                    option.value = doc.document_id;
                    option.textContent = doc.file_name || doc.document_id;
                    pdfSelect.appendChild(option);
                });
            } else {
                pdfsTableBody.innerHTML = '<tr><td colspan="8">No PDFs processed yet.</td></tr>';
            }
        } catch (error) {
            console.error('Error fetching PDFs:', error);
            pdfsTableBody.innerHTML = `<tr><td colspan="8">Error loading PDFs: ${error.message}</td></tr>`;
        }
    }

    // Upload PDF
    if (uploadButton) {
        uploadButton.addEventListener('click', async () => {
            if (!pdfFile.files || pdfFile.files.length === 0) {
                uploadStatus.textContent = 'Please select a PDF file.';
                return;
            }

            const formData = new FormData();
            formData.append('pdf', pdfFile.files[0]);
            uploadStatus.textContent = 'Uploading...';

            try {
                const response = await fetch(`${API_BASE_URL}/upload`, {
                    method: 'POST',
                    body: formData,
                });

                const result = await response.json();

                if (response.ok) {
                    uploadStatus.textContent = `Upload successful: ${result.message} (Document ID: ${result.document_id})`;
                    pdfFile.value = ''; // Clear file input
                    fetchAndListPDFs(); // Refresh PDF list
                } else {
                    uploadStatus.textContent = `Upload failed: ${result.error || 'Unknown error'}`;
                }
            } catch (error) {
                console.error('Error uploading file:', error);
                uploadStatus.textContent = `Upload failed: ${error.message}`;
            }
        });
    }

    // Ask Question
    if (askButton) {
        askButton.addEventListener('click', async () => {
            const query = questionInput.value.trim();
            const selectedPdfId = pdfSelect.value;

            if (!query) {
                answerText.textContent = 'Please enter a question.';
                sourceDocumentsList.innerHTML = '';
                return;
            }

            answerText.textContent = 'Thinking...';
            sourceDocumentsList.innerHTML = '';

            try {
                const requestBody = { query: query };
                if (selectedPdfId) {
                    requestBody.pdf_id = selectedPdfId;
                }

                const response = await fetch(`${API_BASE_URL}/ask`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestBody),
                });

                const result = await response.json();

                if (response.ok) {
                    answerText.textContent = result.answer || 'No answer found.';
                    if (result.source_documents && result.source_documents.length > 0) {
                        result.source_documents.forEach(srcDoc => {
                            const li = document.createElement('li');
                            li.textContent = srcDoc;
                            sourceDocumentsList.appendChild(li);
                        });
                    } else {
                        const li = document.createElement('li');
                        li.textContent = "No specific source documents identified.";
                        sourceDocumentsList.appendChild(li);
                    }
                } else {
                    answerText.textContent = `Error: ${result.error || 'Failed to get answer'}`;
                }
            } catch (error) {
                console.error('Error asking question:', error);
                answerText.textContent = `Error: ${error.message}`;
            }
        });
    }

    // List PDFs button
    if (listPdfsButton) {
        listPdfsButton.addEventListener('click', fetchAndListPDFs);
    }

    // Initial load of PDFs
    fetchAndListPDFs();
});
