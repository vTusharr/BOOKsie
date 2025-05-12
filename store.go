package main

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	_ "github.com/mattn/go-sqlite3"
	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const dbFile = "./pdf_qna.db"
const qdrantCollectionNameConst = "pdf_chunks_collection"
const qdrantVectorSize uint64 = 768

var sqliteDB *sql.DB
var qdrantConn *grpc.ClientConn
var pointsClient qdrant.PointsClient
var collectionsClient qdrant.CollectionsClient

const qdrantHost = "localhost"
const qdrantPort = 6334

// Store struct can be used to group database and Qdrant operations.
type Store struct{}

// NewStore creates a new Store instance.
func NewStore() (*Store, error) {
	return &Store{}, nil
}

type ChunkRecord struct {
	ID             string
	Content        string
	Embedding      []float32
	SourceDocument string
	ChunkIndex     int
}

type ChunkDataToStore struct {
	ID             string
	Content        string
	Embedding      []float32
	SourceDocument string
	ChunkIndex     int
}

type ProcessedDocumentInfo struct {
	ID           string
	FileName     string
	TotalPages   int
	TotalChunks  int
	Title        string
	Author       string
	ProcessedAt  string
	MetadataJSON string
	Status       string // Added Status field
}

func InitDB() error {
	var err error
	dbFilePath := dbFile
	if dbPathEnv := os.Getenv("DB_PATH"); dbPathEnv != "" {
		dbFilePath = dbPathEnv
		log.Printf("Using DB_PATH from environment: %s", dbFilePath)
	} else {
		log.Printf("DB_PATH not set, using default: %s", dbFilePath)
	}

	dbDir := filepath.Dir(dbFilePath)
	if _, err := os.Stat(dbDir); os.IsNotExist(err) {
		log.Printf("Database directory %s does not exist. Creating now...", dbDir)
		if err := os.MkdirAll(dbDir, 0755); err != nil {
			return fmt.Errorf("failed to create database directory %s: %w", dbDir, err)
		}
	}

	_, statErr := os.Stat(dbFilePath)
	sqliteDB, err = sql.Open("sqlite3", dbFilePath+"?_busy_timeout=5000")
	if err != nil {
		return fmt.Errorf("failed to open sqlite database: %w", err)
	}

	_, err = sqliteDB.Exec("PRAGMA journal_mode=WAL;")
	if err != nil {
		log.Printf("warning: failed to set WAL mode for SQLite: %v. Continuing with default.", err)
	} else {
		log.Println("WAL mode enabled for SQLite database.")
	}

	if os.IsNotExist(statErr) {
		log.Printf("SQLite database file '%s' not found, creating new one.", dbFilePath)
	} else {
		log.Printf("Using existing SQLite database file: '%s'", dbFilePath)
	}

	createProcessedDocumentsTableSQL := `CREATE TABLE IF NOT EXISTS processed_documents (
		"source_document_name" TEXT NOT NULL PRIMARY KEY,
		"content_hash" TEXT NOT NULL,
		"processed_at" DATETIME NOT NULL,
		"title" TEXT,
		"author" TEXT,
		"creation_date" TEXT,
		"total_pages" INTEGER,
		"total_chunks" INTEGER,
		"all_metadata_json" TEXT,
		"status" TEXT DEFAULT 'Unknown' -- Added status column
	);`
	_, err = sqliteDB.Exec(createProcessedDocumentsTableSQL)
	if err != nil {
		return fmt.Errorf("failed to create processed_documents table: %w", err)
	}
	log.Println("SQLite database initialized and 'processed_documents' table ensured.")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var qdrantDialAddress string
	envQdrantURL := os.Getenv("QDRANT_URL")

	if envQdrantURL != "" {
		log.Printf("QDRANT_URL from environment: %s", envQdrantURL)

		addrToParse := envQdrantURL
		if strings.HasPrefix(addrToParse, "http://") {
			addrToParse = strings.TrimPrefix(addrToParse, "http://")
		} else if strings.HasPrefix(addrToParse, "https://") {
			addrToParse = strings.TrimPrefix(addrToParse, "https://")
		}

		// Split host and port from the (potentially) scheme-less address
		hostPart := strings.Split(addrToParse, ":")[0]

		if hostPart == "" {
			log.Printf("Host part of QDRANT_URL ('%s') is empty after parsing. Defaulting to 'qdrant:6334'.", envQdrantURL)
			qdrantDialAddress = "qdrant:6334"
		} else {
			// Use the extracted host and the standard gRPC port 6334 for Qdrant
			qdrantDialAddress = fmt.Sprintf("%s:%d", hostPart, 6334) // Qdrant gRPC port
			log.Printf("Derived Qdrant gRPC dial address: %s", qdrantDialAddress)
		}
	} else {
		// This case would be hit if QDRANT_URL is not set in docker-compose for the app service
		log.Printf("QDRANT_URL not set in environment. Using default 'qdrant:6334' for Docker context.")
		qdrantDialAddress = "qdrant:6334"
	}

	log.Printf("Attempting gRPC dial to Qdrant at: %s", qdrantDialAddress)
	qdrantConn, err = grpc.Dial(qdrantDialAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("failed to connect to Qdrant at %s: %w", qdrantDialAddress, err)
	}
	log.Printf("Successfully established gRPC connection to Qdrant at %s", qdrantDialAddress)

	pointsClient = qdrant.NewPointsClient(qdrantConn)
	collectionsClient = qdrant.NewCollectionsClient(qdrantConn)

	collList, err := collectionsClient.List(ctx, &qdrant.ListCollectionsRequest{})
	if err != nil {
		return fmt.Errorf("failed to list Qdrant collections: %w", err)
	}

	collectionExists := false
	for _, coll := range collList.GetCollections() {
		if coll.GetName() == qdrantCollectionNameConst {
			collectionExists = true
			break
		}
	}

	if !collectionExists {
		log.Printf("Qdrant collection '%s' does not exist. Creating now...", qdrantCollectionNameConst)
		_, err = collectionsClient.Create(ctx, &qdrant.CreateCollection{
			CollectionName: qdrantCollectionNameConst,
			VectorsConfig: &qdrant.VectorsConfig{
				Config: &qdrant.VectorsConfig_Params{
					Params: &qdrant.VectorParams{
						Size:     qdrantVectorSize,
						Distance: qdrant.Distance_Cosine,
					},
				},
			},
		})
		if err != nil {
			return fmt.Errorf("failed to create Qdrant collection '%s': %w", qdrantCollectionNameConst, err)
		}
		log.Printf("Qdrant collection '%s' created successfully.", qdrantCollectionNameConst)
	} else {
		log.Printf("Qdrant collection '%s' already exists.", qdrantCollectionNameConst)
	}
	log.Println("Qdrant client initialized and collection ensured.")
	return nil
}

func CloseDBConnections() {
	if sqliteDB != nil {
		err := sqliteDB.Close()
		if err != nil {
			log.Printf("error closing SQLite database: %v", err)
		} else {
			log.Println("SQLite database connection closed.")
		}
	}
	if qdrantConn != nil {
		err := qdrantConn.Close()
		if err != nil {
			log.Printf("error closing Qdrant connection: %v", err)
		} else {
			log.Println("Qdrant connection closed.")
		}
	}
}

func (s *Store) StoreChunksInBatch(chunksToStore []ChunkDataToStore) error {
	if len(chunksToStore) == 0 {
		log.Println("no chunks provided to StoreChunksInBatch")
		return nil
	}
	if pointsClient == nil {
		return fmt.Errorf("qdrant points client is not initialized")
	}

	points := make([]*qdrant.PointStruct, 0, len(chunksToStore))
	sourceDoc := "unknown"
	if len(chunksToStore) > 0 {
		sourceDoc = chunksToStore[0].SourceDocument
	}

	for _, chunk := range chunksToStore {
		if chunk.ID == "" {
			log.Printf("warning: received chunk with empty ID for document %s, chunk index %d. Generating new UUID.", chunk.SourceDocument, chunk.ChunkIndex)
			chunk.ID = uuid.NewString()
		}

		payload := map[string]*qdrant.Value{
			"source_document": {Kind: &qdrant.Value_StringValue{StringValue: chunk.SourceDocument}},
			"content":         {Kind: &qdrant.Value_StringValue{StringValue: chunk.Content}},
			"chunk_index":     {Kind: &qdrant.Value_IntegerValue{IntegerValue: int64(chunk.ChunkIndex)}},
		}

		points = append(points, &qdrant.PointStruct{
			Id:      &qdrant.PointId{PointIdOptions: &qdrant.PointId_Uuid{Uuid: chunk.ID}},
			Payload: payload,
			Vectors: &qdrant.Vectors{VectorsOptions: &qdrant.Vectors_Vector{Vector: &qdrant.Vector{Data: chunk.Embedding}}},
		})
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	waitUpsert := true
	_, err := pointsClient.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: qdrantCollectionNameConst,
		Points:         points,
		Wait:           &waitUpsert,
	})

	if err != nil {
		return fmt.Errorf("failed to upsert points to Qdrant for document '%s': %w", sourceDoc, err)
	}

	log.Printf("Successfully stored %d chunks in batch to Qdrant from document '%s'", len(chunksToStore), sourceDoc)
	return nil
}

type ChunkWithSimilarity struct {
	Chunk      ChunkRecord
	Similarity float32
}

func (s *Store) FindSimilarChunks(queryEmbedding []float32, topN int, pdfIDFilter string) ([]ChunkWithSimilarity, error) {
	if pointsClient == nil {
		return nil, fmt.Errorf("qdrant points client is not initialized")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var qdrantFilter *qdrant.Filter
	if pdfIDFilter != "" {
		log.Printf("FindSimilarChunks: Applying filter for pdf_id: %s", pdfIDFilter)
		qdrantFilter = &qdrant.Filter{
			Must: []*qdrant.Condition{
				{
					ConditionOneOf: &qdrant.Condition_Field{
						Field: &qdrant.FieldCondition{
							Key: "source_document",
							Match: &qdrant.Match{
								MatchValue: &qdrant.Match_Keyword{Keyword: pdfIDFilter},
							},
						},
					},
				},
			},
		}
	} else {
		log.Println("FindSimilarChunks: No pdf_id filter applied.")
	}

	searchRequest := &qdrant.SearchPoints{
		CollectionName: qdrantCollectionNameConst,
		Vector:         queryEmbedding,
		Limit:          uint64(topN),
		Filter:         qdrantFilter,
		WithPayload:    &qdrant.WithPayloadSelector{SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: true}},
	}

	searchResult, err := pointsClient.Search(ctx, searchRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to search Qdrant: %w", err)
	}

	var results []ChunkWithSimilarity
	for _, hit := range searchResult.GetResult() {
		payload := hit.GetPayload()
		contentVal, okContent := payload["content"]
		sourceDocVal, okSourceDoc := payload["source_document"]
		chunkIndexVal, okChunkIndex := payload["chunk_index"]

		if !okContent || !okSourceDoc {
			log.Printf("warning: Qdrant hit ID %s missing 'content' or 'source_document' in payload. Skipping.", hit.GetId().GetUuid())
			continue
		}

		var chunkIndex int
		if okChunkIndex {
			chunkIndex = int(chunkIndexVal.GetIntegerValue())
		} else {
			log.Printf("warning: Qdrant hit ID %s missing 'chunk_index' in payload. Defaulting to 0.", hit.GetId().GetUuid())
		}

		chunkRecord := ChunkRecord{
			ID:             hit.GetId().GetUuid(),
			Content:        contentVal.GetStringValue(),
			SourceDocument: sourceDocVal.GetStringValue(),
			Embedding:      nil,
			ChunkIndex:     chunkIndex,
		}
		results = append(results, ChunkWithSimilarity{
			Chunk:      chunkRecord,
			Similarity: hit.GetScore(),
		})
	}

	return results, nil
}

func GenerateFileHash(pdfData []byte) (string, error) {
	hasher := sha256.New()
	if _, err := hasher.Write(pdfData); err != nil {
		return "", fmt.Errorf("failed to write pdf data to hasher: %w", err)
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func (s *Store) GetProcessedDocumentHash(sourceDocumentName string) (string, error) {
	var contentHash string
	err := sqliteDB.QueryRow("SELECT content_hash FROM processed_documents WHERE source_document_name = ?", sourceDocumentName).Scan(&contentHash)
	if err != nil {
		if err == sql.ErrNoRows {
			return "", nil
		}
		return "", fmt.Errorf("failed to query processed document hash for %s: %w", sourceDocumentName, err)
	}
	return contentHash, nil
}

func (s *Store) MarkDocumentAsProcessed(sourceDocumentName, contentHash string, totalPages, totalChunks int, metadata map[string]interface{}) error {
	log.Printf("Marking document '%s' as processed. Hash: %s, Pages: %d, Chunks: %d", sourceDocumentName, contentHash, totalPages, totalChunks)
	if sqliteDB == nil {
		return fmt.Errorf("database not initialized")
	}

	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		log.Printf("Warning: could not marshal metadata for %s: %v. Storing without all_metadata_json.", sourceDocumentName, err)
		metadataJSON = []byte("{}")
	}

	title, _ := metadata["title"].(string)
	author, _ := metadata["author"].(string)
	creationDate, _ := metadata["creationDate"].(string)

	status := "Processed"
	if totalChunks == 0 && totalPages > 0 { // Assuming if pages > 0 but no chunks, it might be an issue or just no text content
		status = "Processed (No Text Chunks)"
	} else if totalChunks == 0 && totalPages == 0 {
		status = "Processed (Empty or Unreadable)"
	}

	stmt, err := sqliteDB.Prepare("INSERT OR REPLACE INTO processed_documents (source_document_name, content_hash, processed_at, title, author, creation_date, total_pages, total_chunks, all_metadata_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
	if err != nil {
		return fmt.Errorf("failed to prepare statement for marking document processed: %w", err)
	}
	defer stmt.Close()

	_, err = stmt.Exec(sourceDocumentName, contentHash, time.Now().Format(time.RFC3339), title, author, creationDate, totalPages, totalChunks, string(metadataJSON), status)
	if err != nil {
		return fmt.Errorf("failed to execute statement for marking document '%s' processed: %w", sourceDocumentName, err)
	}
	log.Printf("Successfully marked/updated document '%s' as processed in SQLite with status: %s.", sourceDocumentName, status)
	return nil
}

func (s *Store) DeleteChunksBySource(sourceDocumentName string) error {
	log.Printf("Attempting to delete existing chunks for source document: %s", sourceDocumentName)
	if pointsClient == nil {
		return fmt.Errorf("qdrant points client is not initialized")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	_, err := pointsClient.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: qdrantCollectionNameConst,
		Points: &qdrant.PointsSelector{
			PointsSelectorOneOf: &qdrant.PointsSelector_Filter{
				Filter: &qdrant.Filter{
					Must: []*qdrant.Condition{
						{
							ConditionOneOf: &qdrant.Condition_Field{
								Field: &qdrant.FieldCondition{
									Key: "source_document",
									Match: &qdrant.Match{
										MatchValue: &qdrant.Match_Keyword{Keyword: sourceDocumentName},
									},
								},
							},
						},
					},
				},
			},
		},
	})

	if err != nil {
		return fmt.Errorf("failed to delete points for source '%s' from Qdrant: %w", sourceDocumentName, err)
	}

	log.Printf("Successfully submitted request to delete points for source document '%s' from Qdrant.", sourceDocumentName)
	return nil
}

func (s *Store) DeleteProcessedDocumentEntry(sourceDocumentName string) error {
	stmt, err := sqliteDB.Prepare("DELETE FROM processed_documents WHERE source_document_name = ?")
	if err != nil {
		return fmt.Errorf("failed to prepare delete statement for processed_documents entry %s: %w", sourceDocumentName, err)
	}
	defer stmt.Close()

	_, err = stmt.Exec(sourceDocumentName)
	if err != nil {
		return fmt.Errorf("failed to execute delete statement for processed_documents entry %s: %w", sourceDocumentName, err)
	}
	log.Printf("Deleted entry for '%s' from processed_documents table.", sourceDocumentName)
	return nil
}

func (s *Store) GetProcessedDocuments() ([]ProcessedDocumentInfo, error) {
	if sqliteDB == nil {
		return nil, fmt.Errorf("database not initialized")
	}

	rows, err := sqliteDB.Query("SELECT source_document_name, total_pages, total_chunks, title, author, processed_at, all_metadata_json, status FROM processed_documents ORDER BY processed_at DESC")
	if err != nil {
		return nil, fmt.Errorf("failed to query processed_documents: %w", err)
	}
	defer rows.Close()

	var documents []ProcessedDocumentInfo
	for rows.Next() {
		var doc ProcessedDocumentInfo
		var totalPages sql.NullInt64
		var totalChunks sql.NullInt64
		var title sql.NullString
		var author sql.NullString
		var processedAt sql.NullString // Keep as NullString for scanning
		var metadataJSON sql.NullString
		var status sql.NullString // Added for scanning

		if err := rows.Scan(&doc.ID, &totalPages, &totalChunks, &title, &author, &processedAt, &metadataJSON, &status); err != nil {
			log.Printf("Error scanning row from processed_documents: %v", err)
			continue
		}
		doc.FileName = filepath.Base(doc.ID)
		if totalPages.Valid {
			doc.TotalPages = int(totalPages.Int64)
		}
		if totalChunks.Valid {
			doc.TotalChunks = int(totalChunks.Int64)
		}
		if title.Valid {
			doc.Title = title.String
		}
		if author.Valid {
			doc.Author = author.String
		}
		if processedAt.Valid {
			doc.ProcessedAt = processedAt.String
		}
		if metadataJSON.Valid {
			doc.MetadataJSON = metadataJSON.String
		}
		if status.Valid {
			doc.Status = status.String
		} else {
			doc.Status = "Unknown" // Default if NULL in DB
		}

		if (doc.Title == "" || doc.Author == "") && metadataJSON.Valid && metadataJSON.String != "" && metadataJSON.String != "{}" {
			var metadata map[string]interface{}
			if err := json.Unmarshal([]byte(metadataJSON.String), &metadata); err == nil {
				if doc.Title == "" {
					if t, ok := metadata["title"].(string); ok {
						doc.Title = t
					}
				}
				if doc.Author == "" {
					if a, ok := metadata["author"].(string); ok {
						doc.Author = a
					}
				}
			}
		}
		documents = append(documents, doc)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating rows from processed_documents: %w", err)
	}

	return documents, nil
}
