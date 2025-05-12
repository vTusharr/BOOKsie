package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"
)

func firstN(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func main() {
	log.Println("Starting PDF Q&A Service...")

	// Load configuration
	cfg, err := loadConfig("config.json")
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}
	log.Printf("Configuration loaded: %+v", cfg)

	// Initialize database and Qdrant connections
	if err := InitDB(); err != nil {
		log.Fatalf("Failed to initialize DB and Qdrant: %v", err)
	}

	defer CloseDBConnections()

	// Initialize the Qdrant store
	store, err := NewStore() // Assuming NewStore() initializes Qdrant related components internally if needed.
	if err != nil {
		log.Fatalf("Failed to initialize store: %v", err)
	}

	grpcPort := ":50051"
	if portEnv := os.Getenv("GRPC_PORT"); portEnv != "" {
		if _, err := net.LookupPort("tcp", portEnv); err == nil {
			grpcPort = ":" + portEnv
		} else {
			log.Printf("Invalid GRPC_PORT '%s', using default '%s'. Error: %v", portEnv, grpcPort, err)
		}
	}

	lis, err := net.Listen("tcp", grpcPort)
	if err != nil {
		log.Fatalf("Failed to listen on port %s: %v", grpcPort, err)
	}
	log.Printf("gRPC server listening on %s", grpcPort)

	// Example usage: Process a single PDF (replace with actual PDF path)
	// This is just for demonstration; in a real app, you'd get this from an upload or other source.
	if len(os.Args) > 1 {
		pdfPath := os.Args[1]
		log.Printf("Processing PDF specified on command line: %s", pdfPath)
		processSinglePDF(pdfPath, cfg.PythonExecutable, store, cfg) // Pass cfg
	} else {
		log.Println("No PDF path provided on command line. Skipping single PDF processing example.")
	}

	// Increase gRPC server's max receive message size
	maxMsgSize := 100 * 1024 * 1024 // 100 MB
	s := grpc.NewServer(grpc.MaxRecvMsgSize(maxMsgSize))
	// Pass cfg to embeddingServer when registering
	RegisterEmbeddingServiceServer(s, &embeddingServer{config: cfg})
	RegisterQnAServiceServer(s, NewQnaServer(store, cfg))

	reflection.Register(s)
	log.Println("gRPC reflection service registered.")

	go func() {
		log.Println("Starting gRPC server...")
		if err := s.Serve(lis); err != nil {
			log.Printf("gRPC server failed to serve: %v", err)
		}
	}()

	connCtx, connCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer connCancel()
	// Increase gRPC client's max send/receive message size
	grpcClientConn, err := grpc.DialContext(connCtx, "localhost"+grpcPort, grpc.WithInsecure(), grpc.WithBlock(), grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxMsgSize), grpc.MaxCallSendMsgSize(maxMsgSize)))
	if err != nil {
		log.Fatalf("Failed to connect to gRPC server for HTTP frontend: %v", err)
	}
	defer grpcClientConn.Close()
	log.Println("Successfully connected to gRPC server for HTTP frontend.")

	go startHttpServer(grpcClientConn)

	runServerOnlyFlag := flag.Bool("serve", false, "Run only the gRPC and HTTP servers without processing PDFs or asking questions via CLI")
	questionFlag := flag.String("q", "", "Question to ask about the PDF(s)")
	docSourcesFlag := flag.String("sources", "", "Comma-separated list of PDF paths to filter by for the question")
	qnaModeFlag := flag.String("mode", "context_only", "Q&A mode: 'context_only' or 'general_knowledge'")
	flag.Parse()

	pdfPaths := flag.Args()

	if !*runServerOnlyFlag {
		if len(pdfPaths) == 0 && *questionFlag == "" {
			log.Println("No PDF paths or question provided via CLI. Servers are running.")
		}

		if len(pdfPaths) > 0 {
			log.Printf("Processing %d PDF file(s) from command line: %s", len(pdfPaths), strings.Join(pdfPaths, ", "))
			for _, pdfPath := range pdfPaths {
				cleanedPath := filepath.Clean(pdfPath)
				if _, err := os.Stat(cleanedPath); os.IsNotExist(err) {
					log.Printf("Error: PDF file '%s' not found. Skipping.", cleanedPath)
					continue
				}
				log.Printf("Processing PDF from command line: %s", cleanedPath)
				// Pass the configured python executable and the full config to processSinglePDF
				processSinglePDF(cleanedPath, cfg.PythonExecutable, store, cfg) // Pass cfg
			}
		}

		userQuestion := *questionFlag

		if userQuestion != "" {
			log.Printf("\n--- Answering Question (CLI): '%s' ---", userQuestion)

			log.Printf("Generating embedding for the question: '%s'...", userQuestion)
			questionEmbedding, err := GetGeminiEmbedding(userQuestion, cfg.GeminiAPIKey)
			if err != nil {
				log.Fatalf("Failed to generate embedding for question '%s': %v", userQuestion, err)
			}
			log.Printf("Successfully generated embedding for the question.")

			topNChunks := 5
			var pdfIDFilter string
			if *docSourcesFlag != "" {
				docSourcesToFilter := strings.Split(*docSourcesFlag, ",")
				if len(docSourcesToFilter) > 0 {
					pdfIDFilter = filepath.Clean(strings.TrimSpace(docSourcesToFilter[0]))
					log.Printf("Finding top %d similar chunks, filtering by source: %s...", topNChunks, pdfIDFilter)
				}
			} else {
				log.Printf("Finding top %d similar chunks for the question embedding (no source filter)...", topNChunks)
			}

			similarChunkResults, err := store.FindSimilarChunks(questionEmbedding, topNChunks, pdfIDFilter)
			if err != nil {
				log.Fatalf("Failed to find similar chunks: %v", err)
			}

			if len(similarChunkResults) == 0 {
				log.Printf("No relevant chunks found in the database for the question: '%s'. Cannot generate answer.", userQuestion)
			} else {
				var contextChunkContents []string
				log.Printf("Found %d relevant chunks from document(s):", len(similarChunkResults))
				for i, chunkResult := range similarChunkResults {
					contextChunkContents = append(contextChunkContents, chunkResult.Chunk.Content)
					log.Printf("  Chunk %d (Source: %s, Similarity: %.4f): %s...", i+1, chunkResult.Chunk.SourceDocument, chunkResult.Similarity, firstN(chunkResult.Chunk.Content, 70))
				}
				log.Println("Preparing to generate answer...")

				log.Println("Generating answer using Gemini with question and context chunks...")

				var restrictToContextCli bool
				switch strings.ToLower(*qnaModeFlag) {
				case "context_only":
					restrictToContextCli = true
					log.Println("CLI Q&A Mode: Context Only")
				case "general_knowledge":
					restrictToContextCli = false
					log.Println("CLI Q&A Mode: General Knowledge Allowed")
				default:
					log.Printf("Invalid --mode value '%s'. Defaulting to 'context_only'.", *qnaModeFlag)
					restrictToContextCli = true
				}

				answer, errAGFC := GenerateAnswerFromContext(userQuestion, contextChunkContents, cfg.GeminiAPIKey, restrictToContextCli)
				if errAGFC != nil {
					log.Fatalf("Failed to generate answer using Gemini: %v", errAGFC)
				}
				log.Printf("Generated Answer: %s", answer)
				fmt.Printf("Answer: %s\n", answer)

				fmt.Printf("\nQuestion: %s\n", userQuestion)
				fmt.Println("\n--- End of Q&A Example (CLI) ---")
			}
		} else if !*runServerOnlyFlag && len(pdfPaths) > 0 {
			log.Println("No question provided via -q flag, and default question was not triggered. Skipping CLI Q&A.")
		}
	} else {
		log.Println("'-serve' flag detected. Skipping command-line PDF processing and Q&A.")
	}

	log.Println("Application setup complete. gRPC and HTTP servers are running (if not disabled by CLI flags).")
	log.Println("Press Ctrl+C to exit and gracefully shut down the servers.")

	sigs := make(chan os.Signal, 1)
	done := make(chan bool, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigs
		fmt.Println()
		log.Printf("Received signal: %s, initiating graceful shutdown...", sig)

		s.GracefulStop()
		log.Println("gRPC server stopped.")

		log.Println("Closing database connections...")
		CloseDBConnections()
		log.Println("Database connections closed.")
		close(done)
	}()

	<-done
	log.Println("Application exited gracefully.")
}

// processSinglePDF handles the reading, parsing, embedding, and storing of a single PDF file.
// It now accepts the pythonExecutable path and the main config object as arguments.
func processSinglePDF(pdfPath string, pythonExecutable string, store *Store, cfg *Config) { // Added cfg *Config
	originalFileName := filepath.Base(pdfPath)
	log.Printf("Processing PDF: %s (Executable: %s)", originalFileName, pythonExecutable)

	pdfData, err := os.ReadFile(pdfPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("PDF file '%s' not found. Skipping.", pdfPath)
		} else {
			log.Printf("Error reading PDF file '%s': %v", pdfPath, err)
		}
		return
	}

	currentHash, err := GenerateFileHash(pdfData)
	if err != nil {
		log.Printf("Error generating hash for PDF '%s': %v. Skipping.", originalFileName, err)
		return
	}

	storedHash, err := store.GetProcessedDocumentHash(originalFileName)
	if err != nil {
		log.Printf("Error checking processed document status for '%s': %v. Processing anyway.", originalFileName, err)
	} else if storedHash == currentHash {
		log.Printf("PDF '%s' has not changed (hash: %s). Skipping processing.", originalFileName, currentHash)
		return
	} else if storedHash != "" {
		log.Printf("PDF '%s' has changed (new hash: %s, old hash: %s). Reprocessing.", originalFileName, currentHash, storedHash)
		if err := store.DeleteChunksBySource(originalFileName); err != nil {
			log.Printf("Error deleting old chunks for '%s': %v. Proceeding with re-indexing might lead to duplicates or mixed content.", originalFileName, err)
		}
		if err := store.DeleteProcessedDocumentEntry(originalFileName); err != nil {
			log.Printf("Error deleting old processed document entry for '%s': %v.", originalFileName, err)
		}
	} else {
		log.Printf("PDF '%s' is new (hash: %s). Processing.", originalFileName, currentHash)
	}

	log.Printf("Attempting to parse and chunk PDF: %s", originalFileName)
	// Pass the python executable to ParsePDF
	parsedDoc, err := ParsePDF(originalFileName, pdfData, pythonExecutable)
	if err != nil {
		log.Printf("Error parsing PDF %s: %v", originalFileName, err)
		metadataError := map[string]interface{}{
			"status":        "Error: Parsing failed",
			"error_details": err.Error(),
		}
		if errMark := store.MarkDocumentAsProcessed(originalFileName, currentHash, 0, 0, metadataError); errMark != nil {
			log.Printf("Additionally, failed to mark document %s as processed with error status: %v", originalFileName, errMark)
		}
		return
	} else if parsedDoc.Error != "" {
		log.Printf("Error from Python script for PDF %s: %s", originalFileName, parsedDoc.Error)
		metadataError := map[string]interface{}{
			"status":        "Error: Processing script failed",
			"error_details": parsedDoc.Error,
		}
		if errMark := store.MarkDocumentAsProcessed(originalFileName, currentHash, 0, 0, metadataError); errMark != nil {
			log.Printf("Additionally, failed to mark document %s as processed with error status from script: %v", originalFileName, errMark)
		}
		return
	}
	log.Printf("Successfully parsed PDF: %s. Extracted %d characters, %d chunks.", parsedDoc.FileName, len(parsedDoc.Content), len(parsedDoc.Chunks))
	if len(parsedDoc.Metadata) > 0 {
		log.Printf("Extracted Metadata for '%s':", originalFileName)
		for key, value := range parsedDoc.Metadata {
			log.Printf("  %s: %v", key, value)
		}
	}

	numChunks := len(parsedDoc.Chunks)

	totalPages := 0
	if pagesStr, ok := parsedDoc.Metadata["TotalPages"].(string); ok {
		pagesInt, errConv := strconv.Atoi(pagesStr)
		if errConv == nil {
			totalPages = pagesInt
		} else {
			log.Printf("Warning: Could not convert TotalPages metadata '%s' to int for %s: %v", pagesStr, originalFileName, errConv)
		}
	} else if pagesNum, ok := parsedDoc.Metadata["TotalPages"].(float64); ok {
		totalPages = int(pagesNum)
	}

	metadataForStore := make(map[string]interface{})
	for k, v := range parsedDoc.Metadata {
		metadataForStore[k] = v
	}

	if numChunks == 0 {
		log.Printf("No chunks found in '%s' after parsing. Nothing to embed.", originalFileName)
		err := store.MarkDocumentAsProcessed(originalFileName, currentHash, totalPages, 0, metadataForStore)
		if err != nil {
			log.Printf("Error marking document '%s' (with 0 chunks) as processed: %v", originalFileName, err)
		}
		return
	}

	type embeddingResult struct {
		chunkIndex int
		chunkText  string
		embedding  []float32
		err        error
	}

	var (
		chunksToStore []ChunkDataToStore
		results       = make(chan embeddingResult, numChunks)
		wg            sync.WaitGroup
	)

	wg.Add(numChunks)
	log.Printf("Starting concurrent embedding generation for %d chunks from '%s'...", numChunks, originalFileName)

	maxConcurrentEmbeddings := 10
	semaphore := make(chan struct{}, maxConcurrentEmbeddings)

	for i, chunkText := range parsedDoc.Chunks {
		semaphore <- struct{}{}
		go func(idx int, text string) {
			defer wg.Done()
			defer func() { <-semaphore }()

			// Use cfg.GeminiAPIKey passed to processSinglePDF
			embedding, errEmb := GetGeminiEmbedding(text, cfg.GeminiAPIKey)
			if errEmb != nil {
				log.Printf("Error generating embedding for chunk %d from '%s': %v", idx+1, originalFileName, errEmb)
				results <- embeddingResult{chunkIndex: idx, chunkText: text, err: errEmb}
				return
			}
			results <- embeddingResult{chunkIndex: idx, chunkText: text, embedding: embedding, err: nil}
		}(i, chunkText)
	}

	go func() {
		wg.Wait()
		close(results)
		log.Printf("All embedding generation goroutines for '%s' have completed.", originalFileName)
	}()

	processedCount := 0
	failedCount := 0
	progressLogInterval := numChunks / 20
	if progressLogInterval == 0 {
		progressLogInterval = 1
	}

	log.Printf("Collecting embedding results for %d chunks from '%s'. Progress will be logged periodically.", numChunks, originalFileName)

	for result := range results {
		processedCount++
		if result.err != nil {
			failedCount++
		} else {
			chunkID, errID := uuid.NewRandom()
			if errID != nil {
				log.Printf("Error generating UUID for chunk %d from '%s': %v. Skipping this chunk.", result.chunkIndex+1, originalFileName, errID)
				failedCount++
				continue
			}
			chunksToStore = append(chunksToStore, ChunkDataToStore{
				ID:             chunkID.String(),
				SourceDocument: originalFileName,
				Content:        result.chunkText,
				ChunkIndex:     result.chunkIndex,
				Embedding:      result.embedding,
			})
		}
		if processedCount%progressLogInterval == 0 || processedCount == numChunks {
			log.Printf("Embedding progress for '%s': %d/%d chunks processed (%d failed so far).", originalFileName, processedCount, numChunks, failedCount)
		}
	}

	log.Printf("Finished embedding generation for '%s': %d successful, %d failed out of %d chunks.", originalFileName, len(chunksToStore), failedCount, numChunks)

	if len(chunksToStore) > 0 {
		log.Printf("Storing %d embedded chunks for '%s' in batch...", len(chunksToStore), originalFileName)
		if err := store.StoreChunksInBatch(chunksToStore); err != nil {
			log.Printf("CRITICAL: Error storing chunks in batch for '%s': %v. Document will NOT be marked as processed.", originalFileName, err)
			return
		}
		log.Printf("Successfully stored %d chunks for '%s'.", len(chunksToStore), originalFileName)
	} else if numChunks > 0 {
		log.Printf("No chunks to store for '%s' (either all failed embedding or other issues). Document will NOT be marked as processed with this hash.", originalFileName)
		return
	}

	log.Printf("Marking document '%s' (hash: %s) as processed in metadata store.", originalFileName, currentHash)
	err = store.MarkDocumentAsProcessed(originalFileName, currentHash, totalPages, len(chunksToStore), metadataForStore)
	if err != nil {
		log.Printf("Error marking document '%s' as processed: %v", originalFileName, err)
	} else {
		log.Printf("Successfully marked document '%s' (hash: %s) as processed.", originalFileName, currentHash)
	}
	log.Printf("--- Finished processing PDF: %s ---", originalFileName)
}

type embeddingServer struct {
	UnimplementedEmbeddingServiceServer
	config *Config // Added config field
}

type qnaServer struct {
	UnimplementedQnAServiceServer
	dbStore *Store
	config  *Config // Added to store loaded configuration
}

func NewQnaServer(store *Store, cfg *Config) *qnaServer { // Accept *Config
	return &qnaServer{dbStore: store, config: cfg} // Store cfg
}

func (s *embeddingServer) GenerateEmbedding(ctx context.Context, req *GenerateEmbeddingRequest) (*GenerateEmbeddingResponse, error) {
	log.Printf("gRPC GenerateEmbedding called for text: %s...", firstN(req.GetText(), 50))
	if req.GetText() == "" {
		return nil, status.Errorf(codes.InvalidArgument, "Text to embed cannot be empty")
	}

	// Use s.config.GeminiAPIKey as GetGeminiEmbedding now requires it
	embedding, err := GetGeminiEmbedding(req.GetText(), s.config.GeminiAPIKey)
	if err != nil {
		log.Printf("Error generating embedding via Gemini: %v", err)
		return nil, status.Errorf(codes.Internal, "Failed to generate embedding: %v", err)
	}

	log.Printf("Successfully generated embedding for text: %s...", firstN(req.GetText(), 50))
	return &GenerateEmbeddingResponse{Embedding: embedding}, nil
}

func (s *qnaServer) AnswerQuestion(ctx context.Context, req *AnswerQuestionRequest) (*AnswerQuestionResponse, error) {
	log.Printf("gRPC AnswerQuestion called for question: '%s', PDF ID: '%s', Mode: %s", req.GetQuestion(), req.GetPdfId(), req.GetAnswerMode().String())
	if req.GetQuestion() == "" {
		return nil, status.Errorf(codes.InvalidArgument, "Question cannot be empty")
	}

	var sourcesUsed []string
	var contextChunkContents []string

	var restrictToContextGrpc bool
	switch req.GetAnswerMode() {
	case AnswerMode_CONTEXT_ONLY:
		restrictToContextGrpc = true
		log.Println("gRPC Q&A Mode: Context Only")
	case AnswerMode_GENERAL_KNOWLEDGE_ALLOWED:
		restrictToContextGrpc = false
		log.Println("gRPC Q&A Mode: General Knowledge Allowed")
	default:
		log.Printf("Unknown AnswerMode %s. Defaulting to CONTEXT_ONLY.", req.GetAnswerMode().String())
		restrictToContextGrpc = true
	}

	if len(req.GetContextChunks()) > 0 {
		contextChunkContents = req.GetContextChunks()
		log.Printf("Using %d provided context chunks for gRPC AnswerQuestion.", len(contextChunkContents))
	} else {
		log.Printf("No context chunks provided for gRPC AnswerQuestion, attempting to find similar chunks for question: '%s'", req.GetQuestion())
		questionEmbedding, err := GetGeminiEmbedding(req.GetQuestion(), s.config.GeminiAPIKey)
		if err != nil {
			log.Printf("Failed to generate embedding for gRPC question: %v", err)
			return nil, status.Errorf(codes.Internal, "Failed to generate question embedding: %v", err)
		}

		topNChunks := 5
		pdfIDToFilter := req.GetPdfId()
		if pdfIDToFilter != "" {
			log.Printf("Filtering by specific PDF ID for gRPC question: %s", pdfIDToFilter)
		} else {
			log.Printf("No specific PDF ID provided, searching across all documents.")
		}

		similarChunkResults, err := s.dbStore.FindSimilarChunks(questionEmbedding, topNChunks, pdfIDToFilter)
		if err != nil {
			log.Printf("Failed to find similar chunks for gRPC question: %v", err)
			return nil, status.Errorf(codes.Internal, "Failed to find similar chunks: %v", err)
		}
		if len(similarChunkResults) == 0 {
			log.Printf("No relevant chunks found for gRPC question: '%s'", req.GetQuestion())
			if pdfIDToFilter != "" {
				return &AnswerQuestionResponse{Answer: fmt.Sprintf("No relevant information found in document '%s' to answer the question.", pdfIDToFilter), SourceDocuments: []string{}}, nil
			}
			return &AnswerQuestionResponse{Answer: "No relevant information found in any document to answer the question.", SourceDocuments: []string{}}, nil
		}
		for _, chunkResult := range similarChunkResults {
			contextChunkContents = append(contextChunkContents, chunkResult.Chunk.Content)
			found := false
			for _, src := range sourcesUsed {
				if src == chunkResult.Chunk.SourceDocument {
					found = true
					break
				}
			}
			if !found {
				sourcesUsed = append(sourcesUsed, chunkResult.Chunk.SourceDocument)
			}
		}
		log.Printf("Found %d relevant chunks to answer gRPC question.", len(contextChunkContents))
	}

	if len(contextChunkContents) == 0 && restrictToContextGrpc {
		log.Printf("No context available to answer gRPC question: '%s' (mode: CONTEXT_ONLY)", req.GetQuestion())
		var noContextMessage string
		if req.GetPdfId() != "" {
			noContextMessage = fmt.Sprintf("No relevant information found in document '%s' to answer the question.", req.GetPdfId())
		} else {
			noContextMessage = "No relevant information found in any document to answer the question."
		}
		return &AnswerQuestionResponse{Answer: noContextMessage, SourceDocuments: []string{}}, nil
	}

	// Use s.config.GeminiAPIKey as GenerateAnswerFromContext requires it
	answer, err := GenerateAnswerFromContext(req.GetQuestion(), contextChunkContents, s.config.GeminiAPIKey, restrictToContextGrpc)
	if err != nil {
		log.Printf("Error generating answer via Gemini for gRPC call: %v", err)
		return nil, status.Errorf(codes.Internal, "Failed to generate answer: %v", err)
	}

	log.Printf("Successfully generated answer for gRPC question: '%s'", req.GetQuestion())
	return &AnswerQuestionResponse{Answer: answer, SourceDocuments: sourcesUsed}, nil
}

func (s *qnaServer) UploadPDF(ctx context.Context, req *UploadPDFRequest) (*UploadPDFResponse, error) {
	fileName := req.GetFileName()
	pdfContent := req.GetPdfContent()
	log.Printf("gRPC UploadPDF called for file: %s, size: %d bytes", fileName, len(pdfContent))

	if fileName == "" {
		return nil, status.Errorf(codes.InvalidArgument, "File name cannot be empty")
	}
	if len(pdfContent) == 0 {
		return nil, status.Errorf(codes.InvalidArgument, "PDF content cannot be empty")
	}

	uploadDir := "./uploads"
	if err := os.MkdirAll(uploadDir, os.ModePerm); err != nil {
		log.Printf("Failed to create upload directory '%s': %v", uploadDir, err)
		return nil, status.Errorf(codes.Internal, "Failed to create upload directory")
	}
	saneFileName := filepath.Base(fileName)
	if saneFileName == "." || saneFileName == "/" || saneFileName == "\\" {
		saneFileName = "uploaded_file.pdf"
		log.Printf("Original filename '%s' was problematic, using '%s'", fileName, saneFileName)
	}
	tempPdfPath := filepath.Join(uploadDir, saneFileName)

	err := os.WriteFile(tempPdfPath, pdfContent, 0644)
	if err != nil {
		log.Printf("Failed to write uploaded PDF to temporary file '%s': %v", tempPdfPath, err)
		return nil, status.Errorf(codes.Internal, "Failed to save uploaded PDF")
	}
	log.Printf("Uploaded PDF temporarily saved to: %s", tempPdfPath)

	go func() {
		log.Printf("Processing PDF in background: %s (Original: %s)", tempPdfPath, fileName)
		// Pass s.config (which is *Config) to processSinglePDF
		processSinglePDF(tempPdfPath, s.config.PythonExecutable, s.dbStore, s.config)

		if err := os.Remove(tempPdfPath); err != nil {
			log.Printf("Warning: Failed to remove temporary uploaded file %s: %v", tempPdfPath, err)
		} else {
			log.Printf("Successfully removed temporary uploaded file %s", tempPdfPath)
		}
	}()

	docID := saneFileName
	log.Printf("PDF '%s' (ID: %s) submitted for processing via gRPC.", fileName, docID)

	return &UploadPDFResponse{
		Message:    fmt.Sprintf("PDF '%s' received and submitted for background processing.", fileName),
		DocumentId: docID,
	}, nil
}

func (s *qnaServer) ListProcessedPDFs(ctx context.Context, req *ListProcessedPDFsRequest) (*ListProcessedPDFsResponse, error) {
	log.Println("gRPC ListProcessedPDFs called")

	documents, err := s.dbStore.GetProcessedDocuments()
	if err != nil {
		log.Printf("Error getting processed documents from DB: %v", err)
		return nil, status.Errorf(codes.Internal, "Failed to retrieve list of processed documents: %v", err)
	}

	var pbDocuments []*PdfDocumentMetadata
	for _, doc := range documents {
		pbDoc := &PdfDocumentMetadata{
			DocumentId:   doc.ID,
			FileName:     doc.FileName,
			Title:        doc.Title,
			Author:       doc.Author,
			TotalPages:   int32(doc.TotalPages),
			TotalChunks:  int32(doc.TotalChunks),
			ProcessedAt:  doc.ProcessedAt,
			Status:       doc.Status,
			MetadataJson: doc.MetadataJSON,
		}
		pbDocuments = append(pbDocuments, pbDoc)
	}

	log.Printf("Successfully retrieved %d processed documents.", len(pbDocuments))
	return &ListProcessedPDFsResponse{Documents: pbDocuments}, nil
}

func startHttpServer(grpcConn *grpc.ClientConn) {
	httpPort := os.Getenv("HTTP_PORT")
	if httpPort == "" {
		httpPort = "8080"
	}

	mux := http.NewServeMux()

	staticDir := "./static/"
	fs := http.FileServer(http.Dir(staticDir))
	mux.Handle("/", fs)

	qnaClient := NewQnAServiceClient(grpcConn)

	mux.HandleFunc("/upload", handlePdfUpload(qnaClient))
	mux.HandleFunc("/ask", handleAskQuestion(qnaClient))
	mux.HandleFunc("/pdfs", handleListPdfs(qnaClient))

	log.Printf("HTTP server starting on port %s", httpPort)
	if err := http.ListenAndServe(":"+httpPort, mux); err != nil {
		log.Fatalf("HTTP server failed to start: %v", err)
	}
}

func handlePdfUpload(client QnAServiceClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
			return
		}

		// Changed "pdfFile" to "pdf" to match the frontend
		file, header, err := r.FormFile("pdf")
		if err != nil {
			log.Printf("Error retrieving file from form-data: %v", err)
			http.Error(w, "Error retrieving file from form-data: "+err.Error(), http.StatusBadRequest)
			return
		}
		defer file.Close()

		fileBytes, err := io.ReadAll(file)
		if err != nil {
			log.Printf("Error reading file content: %v", err)
			http.Error(w, "Error reading file content: "+err.Error(), http.StatusInternalServerError)
			return
		}

		log.Printf("Received file upload via HTTP: %s, size: %d", header.Filename, len(fileBytes))

		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()

		resp, err := client.UploadPDF(ctx, &UploadPDFRequest{
			FileName:   header.Filename,
			PdfContent: fileBytes,
		})
		if err != nil {
			log.Printf("gRPC UploadPDF call failed: %v", err)
			// Check if it's a gRPC status error
			st, ok := status.FromError(err)
			if ok {
				http.Error(w, "gRPC error: "+st.Message(), http.StatusInternalServerError)
			} else {
				http.Error(w, "Failed to upload PDF: "+err.Error(), http.StatusInternalServerError)
			}
			return
		}

		log.Printf("gRPC UploadPDF call successful: %s, DocID: %s", resp.Message, resp.DocumentId)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func handleAskQuestion(client QnAServiceClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
			return
		}

		var reqData struct {
			Query string `json:"query"`
			PdfId string `json:"pdf_id,omitempty"`
			Mode  string `json:"mode,omitempty"`
		}

		if err := json.NewDecoder(r.Body).Decode(&reqData); err != nil {
			log.Printf("Error decoding /ask request body: %v", err)
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		if reqData.Query == "" {
			http.Error(w, "Query cannot be empty", http.StatusBadRequest)
			return
		}

		log.Printf("Received /ask request via HTTP: Query='%s', PdfId='%s', Mode='%s'", reqData.Query, reqData.PdfId, reqData.Mode)

		grpcAnswerMode := AnswerMode_CONTEXT_ONLY
		if strings.ToLower(reqData.Mode) == "general_knowledge" {
			grpcAnswerMode = AnswerMode_GENERAL_KNOWLEDGE_ALLOWED
		}

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		grpcReq := &AnswerQuestionRequest{
			Question:   reqData.Query,
			PdfId:      reqData.PdfId,
			AnswerMode: grpcAnswerMode,
		}

		resp, err := client.AnswerQuestion(ctx, grpcReq)
		if err != nil {
			log.Printf("gRPC AnswerQuestion call failed: %v", err)
			st, ok := status.FromError(err)
			if ok {
				http.Error(w, "gRPC error: "+st.Message(), http.StatusInternalServerError)
			} else {
				http.Error(w, "Failed to ask question: "+err.Error(), http.StatusInternalServerError)
			}
			return
		}

		log.Printf("gRPC AnswerQuestion call successful for query: '%s'", reqData.Query)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func handleListPdfs(client QnAServiceClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Only GET method is allowed", http.StatusMethodNotAllowed)
			return
		}
		log.Println("Received /pdfs request via HTTP")

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		resp, err := client.ListProcessedPDFs(ctx, &ListProcessedPDFsRequest{})
		if err != nil {
			log.Printf("gRPC ListProcessedPDFs call failed: %v", err)
			st, ok := status.FromError(err)
			if ok {
				http.Error(w, "gRPC error: "+st.Message(), http.StatusInternalServerError)
			} else {
				http.Error(w, "Failed to list PDFs: "+err.Error(), http.StatusInternalServerError)
			}
			return
		}

		log.Printf("gRPC ListProcessedPDFs call successful, found %d documents.", len(resp.Documents))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

type Config struct {
	GeminiAPIKey       string `json:"geminiApiKey"`
	QdrantURL          string `json:"qdrantUrl"` // Added QdrantURL field
	DBPath             string `json:"dbPath"`
	PythonExecutable   string `json:"pythonExecutable"` // Corrected field name and JSON tag
	PDFProcessorScript string `json:"pdfProcessorScript"`
}

func loadConfig(filePath string) (*Config, error) {
	log.Printf("Loading configuration from: %s", filePath)
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", filePath, err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config JSON: %w", err)
	}

	if cfg.GeminiAPIKey == "" {
		log.Println("Warning: GeminiAPIKey is not set in the config.")
	}
	if cfg.QdrantURL == "" {
		log.Println("Warning: QdrantURL is not set in the config.")
	}
	if cfg.DBPath == "" {
		log.Println("Warning: DBPath is not set in the config.")
	}
	if cfg.PythonExecutable == "" { // Corrected field name usage
		log.Println("Warning: PythonExecutable is not set in the config. Using default 'python'.")
		cfg.PythonExecutable = "python" // Corrected field name usage
	}
	if cfg.PDFProcessorScript == "" {
		log.Println("Warning: PDFProcessorScript is not set in the config.")
	}
	return &cfg, nil
}
