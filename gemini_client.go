package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"strings"
	"time"
)

// --- Gemini Specific Constants ---
const geminiEmbeddingsAPIURLFormat = "https://generativelanguage.googleapis.com/v1beta/models/%s:embedContent"
const geminiChatCompletionsAPIURLFormat = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"

// TODO: Define Gemini model names - Ensure these are the correct and available model identifiers.
const geminiTextModel = "gemini-2.5-flash-latest" // Changed to gemini-1.5-flash-latest
const geminiEmbeddingModel = "text-embedding-004" // Example, verify with Gemini documentation

const ( // Retry constants
	maxRetries     = 5
	initialBackoff = 1 * time.Second
	maxBackoff     = 60 * time.Second
	backoffFactor  = 2
	jitterFactor   = 0.1
)

// Helper function to get first N characters of a string for logging (local to this file)
func geminiClientFirstN(s string, n int) string {
	if len(s) <= n {
		return s
	}
	r := []rune(s)
	if len(r) < n {
		n = len(r)
	}
	return string(r[:n])
}

// --- Gemini Embedding Structures ---
// Based on: https://ai.google.dev/docs/reference/rest/v1beta/models/embedContent
type GeminiEmbedContentRequest struct {
	Model                string  `json:"model"` // e.g., "models/text-embedding-004"
	Content              Content `json:"content"`
	TaskType             string  `json:"task_type,omitempty"`            // Optional: e.g., "RETRIEVAL_DOCUMENT"
	Title                string  `json:"title,omitempty"`                // Optional
	OutputDimensionality int     `json:"outputDimensionality,omitempty"` // Optional
}

type Content struct {
	Parts []Part `json:"parts"`
	Role  string `json:"role,omitempty"` // Optional: e.g., "user"
}

type Part struct {
	Text string `json:"text"`
	// InlineData *Blob `json:"inline_data,omitempty"` // For multimodal
}

type GeminiEmbedContentResponse struct {
	Embedding *ContentEmbedding `json:"embedding,omitempty"`
	// Error     *GeminiAPIError      `json:"error,omitempty"` // Define GeminiAPIError struct if needed for detailed errors
}

type ContentEmbedding struct {
	Values []float32 `json:"values"`
}

// --- Gemini Chat Completions Structures ---
// TODO: Define appropriate structs for Gemini chat completions API requests and responses
// Based on: https://ai.google.dev/docs/reference/rest/v1beta/models/generateContent
type GeminiGenerateContentRequest struct {
	Contents         []Content         `json:"contents"`
	SafetySettings   []SafetySetting   `json:"safetySettings,omitempty"`
	GenerationConfig *GenerationConfig `json:"generationConfig,omitempty"`
	Tools            []Tool            `json:"tools,omitempty"`
}

type SafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

type GenerationConfig struct {
	Temperature     *float32 `json:"temperature,omitempty"`
	TopP            *float32 `json:"topP,omitempty"`
	TopK            *int     `json:"topK,omitempty"`
	CandidateCount  *int     `json:"candidateCount,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations,omitempty"`
}

type FunctionDeclaration struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// Parameters  *Schema `json:"parameters,omitempty"` // Define Schema if using function calling
}

type GeminiGenerateContentResponse struct {
	Candidates     []Candidate    `json:"candidates,omitempty"`
	PromptFeedback PromptFeedback `json:"promptFeedback,omitempty"`
	// Error          *GeminiError   `json:"error,omitempty"` // Define GeminiError struct if needed
}

type Candidate struct {
	Content       Content        `json:"content"`
	FinishReason  string         `json:"finishReason,omitempty"` // e.g., "STOP", "MAX_TOKENS", "SAFETY"
	SafetyRatings []SafetyRating `json:"safetyRatings,omitempty"`
	// CitationMetadata CitationMetadata `json:"citationMetadata,omitempty"`
	Index int `json:"index"`
}

type PromptFeedback struct {
	SafetyRatings []SafetyRating `json:"safetyRatings,omitempty"`
	BlockReason   string         `json:"blockReason,omitempty"`
}

type SafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"` // e.g., "NEGLIGIBLE", "LOW", "MEDIUM", "HIGH"
	// Blocked     bool   `json:"blocked,omitempty"`
}

// GetGeminiEmbedding sends a request to the Gemini API to get embeddings for a given text.
// It now requires the Gemini API key to be passed as a parameter.
func GetGeminiEmbedding(text string, geminiAPIKey string) ([]float32, error) {
	// Ensure API key is available
	if geminiAPIKey == "" {
		log.Println("Error: Gemini API key is empty in GetGeminiEmbedding.")
		return nil, fmt.Errorf("Gemini API key is missing")
	}

	// The model name for the request body should typically be in the format "models/MODEL_ID"
	// The model name for the URL is just "MODEL_ID"
	requestModelID := fmt.Sprintf("models/%s", geminiEmbeddingModel)
	urlModelID := geminiEmbeddingModel

	url := fmt.Sprintf(geminiEmbeddingsAPIURLFormat, urlModelID) + "?key=" + geminiAPIKey

	reqPayload := GeminiEmbedContentRequest{
		Model: requestModelID, // Model identifier for the request body
		Content: Content{
			Parts: []Part{{Text: text}},
			// Role: "user", // Optional, can be omitted
		},
		TaskType: "RETRIEVAL_DOCUMENT", // Good default for Q&A context retrieval
	}

	payloadBytes, err := json.Marshal(reqPayload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Gemini embedding request: %w", err)
	}

	var lastErr error
	currentBackoff := initialBackoff

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			// Apply jitter: random percentage of currentBackoff
			jitter := time.Duration(rand.Float64() * jitterFactor * float64(currentBackoff))
			time.Sleep(currentBackoff + jitter)
			currentBackoff *= time.Duration(backoffFactor)
			if currentBackoff > maxBackoff {
				currentBackoff = maxBackoff
			}
		}

		log.Printf("Attempt %d: Calling Gemini Embedding API for text: %s...", attempt+1, geminiClientFirstN(text, 30))

		req, err := http.NewRequest("POST", url, bytes.NewBuffer(payloadBytes))
		if err != nil {
			lastErr = fmt.Errorf("failed to create new HTTP request for Gemini embedding (attempt %d): %w", attempt+1, err)
			continue // Retry
		}
		req.Header.Set("Content-Type", "application/json")

		httpClient := &http.Client{Timeout: 30 * time.Second} // Reasonable timeout
		resp, err := httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("failed to execute Gemini embedding request (attempt %d): %w", attempt+1, err)
			// Network errors are generally retryable
			continue
		}

		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			lastErr = fmt.Errorf("failed to read Gemini embedding response body (attempt %d, status %s): %w", attempt+1, resp.Status, readErr)
			// Potentially retryable if it was a transient issue, but could also indicate a problem with the response itself.
			// For simplicity, we'll retry on read errors too.
			continue
		}

		if resp.StatusCode == http.StatusOK {
			var embedResp GeminiEmbedContentResponse
			if err := json.Unmarshal(bodyBytes, &embedResp); err != nil {
				return nil, fmt.Errorf("failed to unmarshal Gemini embedding success response (status %s): %w. Body: %s", resp.Status, err, string(bodyBytes))
			}
			if embedResp.Embedding != nil && len(embedResp.Embedding.Values) > 0 {
				log.Printf("Successfully got embedding from Gemini. Vector length: %d", len(embedResp.Embedding.Values))
				return embedResp.Embedding.Values, nil
			}
			// Successful status but empty or malformed embedding
			return nil, fmt.Errorf("Gemini API returned status OK but no valid embedding found. Body: %s", string(bodyBytes))
		}

		// Handle non-OK status codes
		errorMsg := fmt.Sprintf("Gemini API error (attempt %d): %s (status %d). Body: %s", attempt+1, resp.Status, resp.StatusCode, string(bodyBytes))
		log.Println(errorMsg)
		lastErr = fmt.Errorf(errorMsg)

		// Retry on 5xx server errors or 429 (Too Many Requests)
		if resp.StatusCode >= 500 || resp.StatusCode == http.StatusTooManyRequests {
			log.Printf("Retrying due to status code %d...", resp.StatusCode)
			continue
		} else {
			// Non-retryable client error (e.g., 400, 401, 403, 404)
			return nil, lastErr
		}
	}

	log.Printf("Failed to get embedding after %d attempts for text: %s...", maxRetries, geminiClientFirstN(text, 30))
	return nil, fmt.Errorf("failed to get Gemini embedding after %d retries: %w", maxRetries, lastErr)
}

// GenerateAnswerFromContext generates an answer using Gemini based on the question and provided context.
// restrictToContext determines if the LLM should only use the provided context or can use its general knowledge.
func GenerateAnswerFromContext(question string, context []string, geminiAPIKey string, restrictToContext bool) (string, error) {
	if geminiAPIKey == "" {
		log.Println("Error: Gemini API key is empty in GenerateAnswerFromContext.")
		return "", fmt.Errorf("Gemini API key is missing")
	}

	// The model name for the URL is just "MODEL_ID"
	urlModelID := geminiTextModel
	url := fmt.Sprintf(geminiChatCompletionsAPIURLFormat, urlModelID) + "?key=" + geminiAPIKey

	// Construct the prompt
	var promptBuilder strings.Builder
	if restrictToContext {
		promptBuilder.WriteString("You are a helpful assistant. Answer the following question based *only* on the provided context. " +
			"If the answer is not found in the context, say 'The answer is not found in the provided documents.' " +
			"For each piece of information you use from the context, cite the specific context snippet number it came from using the format [snippet N].\n\nContext:\n")
	} else {
		promptBuilder.WriteString("You are a helpful assistant. Answer the following question. You may use your general knowledge. " +
			"If you use information from the provided context snippets (if any), please cite the specific context snippet number it came from using the format [snippet N]. " +
			"If the context is empty or not relevant, answer using your general knowledge. If you cannot answer the question even with general knowledge, say that you cannot provide an answer.\n\nContext (if provided):\n")
	}

	if len(context) > 0 {
		for i, ctx := range context {
			promptBuilder.WriteString(fmt.Sprintf("Snippet %d: %s\n", i+1, ctx))
			// Limit context in prompt if too long - this logic might need adjustment based on overall prompt limits
			if i > 5 && len(context) > 10 {
				promptBuilder.WriteString(fmt.Sprintf("... (and %d more context snippets)\n", len(context)-(i+1)))
				break
			}
		}
	} else if restrictToContext {
		// If restricted to context and no context is given, it's an issue for the prompt.
		promptBuilder.WriteString("No context was provided. You must answer based only on provided context.\n")
	} else {
		promptBuilder.WriteString("No specific context provided. Please answer using your general knowledge.\n")
	}

	promptBuilder.WriteString(fmt.Sprintf("\nQuestion: %s\n\nAnswer:", question))
	fullPrompt := promptBuilder.String()

	// Prepare request payload
	reqPayload := GeminiGenerateContentRequest{
		Contents: []Content{
			{
				Parts: []Part{{Text: fullPrompt}},
				Role:  "user",
			},
		},
		GenerationConfig: &GenerationConfig{
			Temperature:     refFloat32(0.7),
			MaxOutputTokens: refInt(1024),
		},
	}

	payloadBytes, err := json.Marshal(reqPayload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal Gemini chat request: %w", err)
	}

	var lastErr error
	currentBackoff := initialBackoff

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			jitter := time.Duration(rand.Float64() * jitterFactor * float64(currentBackoff))
			time.Sleep(currentBackoff + jitter)
			currentBackoff *= time.Duration(backoffFactor)
			if currentBackoff > maxBackoff {
				currentBackoff = maxBackoff
			}
		}

		log.Printf("Attempt %d: Calling Gemini Chat API. Prompt: %s...", attempt+1, geminiClientFirstN(fullPrompt, 70))

		req, err := http.NewRequest("POST", url, bytes.NewBuffer(payloadBytes))
		if err != nil {
			lastErr = fmt.Errorf("failed to create new HTTP request for Gemini chat (attempt %d): %w", attempt+1, err)
			continue
		}
		req.Header.Set("Content-Type", "application/json")

		httpClient := &http.Client{Timeout: 90 * time.Second}
		resp, err := httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("failed to execute Gemini chat request (attempt %d): %w", attempt+1, err)
			continue
		}

		bodyBytes, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			lastErr = fmt.Errorf("failed to read Gemini chat response body (attempt %d, status %s): %w", attempt+1, resp.Status, readErr)
			continue
		}

		if resp.StatusCode == http.StatusOK {
			var chatResp GeminiGenerateContentResponse
			if err := json.Unmarshal(bodyBytes, &chatResp); err != nil {
				return "", fmt.Errorf("failed to unmarshal Gemini chat success response (status %s): %w. Body: %s", resp.Status, err, string(bodyBytes))
			}

			// Check for prompt-level blocking
			if chatResp.PromptFeedback.BlockReason != "" {
				errMsg := fmt.Sprintf("Gemini prompt was blocked. Reason: %s.", chatResp.PromptFeedback.BlockReason)
				log.Println(errMsg)
				for _, rating := range chatResp.PromptFeedback.SafetyRatings {
					if rating.Probability != "NEGLIGIBLE" { // Log non-negligible ratings
						log.Printf("Prompt Safety Rating - Category: %s, Probability: %s", rating.Category, rating.Probability)
					}
				}
				return "", fmt.Errorf(errMsg)
			}

			if len(chatResp.Candidates) > 0 {
				candidate := chatResp.Candidates[0]

				// Check for candidate-level blocking or issues
				if candidate.FinishReason == "SAFETY" {
					errMsg := "Gemini response was blocked due to safety concerns."
					log.Println(errMsg)
					for _, rating := range candidate.SafetyRatings {
						// Log all safety ratings that led to the block
						log.Printf("Candidate Safety Rating - Category: %s, Probability: %s", rating.Category, rating.Probability)
					}
					return "", fmt.Errorf(errMsg)
				}

				if candidate.FinishReason == "MAX_TOKENS" {
					log.Printf("Warning: Gemini response was truncated due to maximum token limit. Consider increasing MaxOutputTokens or refining the prompt/context.")
				}

				if candidate.FinishReason == "RECITATION" {
					log.Printf("Warning: Gemini response was blocked due to recitation policy. This may happen if the output is too similar to copyrighted material.")
				}

				if candidate.FinishReason == "OTHER" {
					log.Printf("Warning: Gemini response finished due to an 'OTHER' reason. This may indicate an unexpected issue.")
				}

				// Log any non-negligible safety ratings even if not blocked
				for _, rating := range candidate.SafetyRatings {
					if rating.Probability != "NEGLIGIBLE" && rating.Probability != "LOW" { // Stricter logging for candidate content
						log.Printf("Warning: Gemini response candidate has safety rating - Category: %s, Probability: %s", rating.Category, rating.Probability)
					}
				}

				if len(candidate.Content.Parts) > 0 {
					answerText := candidate.Content.Parts[0].Text
					log.Printf("Successfully got answer from Gemini. Finish reason: %s", candidate.FinishReason)
					return answerText, nil
				}
			}
			// If no candidates or no parts in the first candidate
			return "", fmt.Errorf("Gemini API returned status OK but no valid answer content found. Body: %s", string(bodyBytes))
		}

		errorMsg := fmt.Sprintf("Gemini Chat API error (attempt %d): %s (status %d). Body: %s", attempt+1, resp.Status, resp.StatusCode, string(bodyBytes))
		log.Println(errorMsg)
		lastErr = fmt.Errorf(errorMsg)

		if resp.StatusCode >= 500 || resp.StatusCode == http.StatusTooManyRequests {
			log.Printf("Retrying due to status code %d...", resp.StatusCode)
			continue
		} else {
			return "", lastErr
		}
	}

	log.Printf("Failed to get answer after %d attempts for prompt: %s...", maxRetries, geminiClientFirstN(fullPrompt, 70))
	return "", fmt.Errorf("failed to get Gemini answer after %d retries: %w", maxRetries, lastErr)
}

// Helper functions to get pointers for optional fields in GenerationConfig
func refFloat32(f float32) *float32 {
	return &f
}

func refInt(i int) *int {
	return &i
}
