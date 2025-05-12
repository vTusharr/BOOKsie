package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

// ParsedDocument holds the content and metadata of a parsed PDF.
// The fields are tagged for JSON unmarshalling from the Python script's output.
type ParsedDocument struct {
	FileName string                 `json:"FileName"`
	Content  string                 `json:"Content"`
	Chunks   []string               `json:"Chunks"`
	Metadata map[string]interface{} `json:"Metadata"`        // Added to capture PDF metadata
	Error    string                 `json:"Error,omitempty"` // Captures errors from the Python script
}

// ParsePDF uses an external Python script (PyMuPDF) to extract text and chunk a PDF.
// It now accepts pythonInterpreter as an argument.
func ParsePDF(originalFileName string, pdfData []byte, pythonInterpreter string) (*ParsedDocument, error) {
	log.Printf("Starting to parse PDF: %s using Python script (PyMuPDF)", originalFileName)

	// Determine the Python interpreter command
	interpreterCmd := pythonInterpreter
	if interpreterCmd == "" {
		log.Println("Python interpreter not provided, defaulting to 'python3'")
		interpreterCmd = "python3"
	}

	// Create a temporary file for the PDF data
	tempDir := os.TempDir()
	tempFile, err := ioutil.TempFile(tempDir, "upload-*.pdf")
	if err != nil {
		return nil, fmt.Errorf("failed to create temporary PDF file: %w", err)
	}
	defer func() {
		tempFile.Close()                                   // Close the file first
		if err := os.Remove(tempFile.Name()); err != nil { // Then remove it
			log.Printf("Warning: failed to remove temporary PDF file %s: %v", tempFile.Name(), err)
		}
	}()

	if _, err := tempFile.Write(pdfData); err != nil {
		return nil, fmt.Errorf("failed to write PDF data to temporary file %s: %w", tempFile.Name(), err)
	}
	if err := tempFile.Close(); err != nil { // Close after writing, before passing to script
		return nil, fmt.Errorf("failed to close temporary PDF file %s after writing: %w", tempFile.Name(), err)
	}

	// Get the absolute path to pdf_processor.py, assuming it's in the same directory as the Go executable
	exePath, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("failed to get executable path: %w", err)
	}
	scriptPath := filepath.Join(filepath.Dir(exePath), "pdf_processor.py")

	// Check if the script exists
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		// Fallback for when running with `go run .` where executable is in a temp dir
		wd, err := os.Getwd()
		if err != nil {
			return nil, fmt.Errorf("failed to get current working directory: %w", err)
		}
		scriptPath = filepath.Join(wd, "pdf_processor.py")
		if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
			return nil, fmt.Errorf("pdf_processor.py not found at %s or in executable directory", scriptPath)
		}
	}

	cmd := exec.Command(interpreterCmd, scriptPath, tempFile.Name(), originalFileName)
	log.Printf("Executing command: %s", cmd.String())

	output, err := cmd.CombinedOutput() // CombinedOutput captures both stdout and stderr
	if err != nil {
		// If Python script itself fails to execute or returns non-zero exit code
		log.Printf("Error executing python script for %s. Output: %s", originalFileName, string(output))
		return nil, fmt.Errorf("python script execution failed for %s: %w. Output: %s", originalFileName, err, string(output))
	}

	var parsedDoc ParsedDocument
	if err := json.Unmarshal(output, &parsedDoc); err != nil {
		log.Printf("Error unmarshalling JSON from python script for %s. Output: %s", originalFileName, string(output))
		return nil, fmt.Errorf("failed to unmarshal JSON from python script for %s: %w. Raw output: %s", originalFileName, err, string(output))
	}

	// Check if the Python script reported an error internally (e.g., PyMuPDF error)
	if parsedDoc.Error != "" {
		log.Printf("Python script reported an error for %s: %s", originalFileName, parsedDoc.Error)
		return &parsedDoc, fmt.Errorf("error during PDF processing in python script for %s: %s", originalFileName, parsedDoc.Error)
	}

	log.Printf("Successfully parsed and chunked %s using Python script. Chunks: %d", originalFileName, len(parsedDoc.Chunks))
	return &parsedDoc, nil
}
