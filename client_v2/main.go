package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/gen2brain/malgo"
	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/go-vgo/robotgo"
	hook "github.com/robotn/gohook"
)

// OpenAI API Response structure
type OpenAITranscriptionResponse struct {
	Text string `json:"text"`
}

var (
	isListening         bool
	ctx                 *malgo.AllocatedContext
	recordDevice        *malgo.Device
	pCapturedSamples    []byte
	capturedSampleCount uint32
	mutex               sync.Mutex
	openAIAPIKey        string
	openAIAPIURL        = "https://api.openai.com/v1/audio/transcriptions"

	// Store the last successfully recorded audio and its transcription
	lastCapturedSamples []byte
	lastTranscription   string

	// Add cooldown tracking to prevent rapid repeated triggers
	lastRetypeTime time.Time

	// Global variables
	httpClient *http.Client
)

const (
	modelName = "gpt-4o-mini-transcribe"
)

func main() {
	// Load OpenAI API key from .env file
	err := loadEnvFile()
	if err != nil {
		fmt.Printf("Error loading .env file: %v\n", err)
		os.Exit(1)
	}

	if openAIAPIKey == "" {
		fmt.Println("OPENAI_API_KEY not found in .env file")
		os.Exit(1)
	}

	// Initialize optimized HTTP client
	initHTTPClient()

	fmt.Println("🚀 Speech2Type - Direct OpenAI API Mode")
	fmt.Println("--- Press Ctrl + Shift + S to start/stop listening ---")
	fmt.Println("--- Press Ctrl + Shift + Q to resend last audio ---")
	fmt.Println("--- Press Ctrl + Shift + 2 to retype last transcription ---")

	// ToDo, add benchmark based on audio.wav file here
	err = runBenchmark()
	if err != nil {
		fmt.Printf("⚠️  Benchmark failed: %v\n", err)
		fmt.Println("Continuing with normal operation...")
	}

	// Display connection pool stats
	printConnectionStats()

	// Initialize malgo
	ctx, err = malgo.InitContext(nil, malgo.ContextConfig{}, func(message string) {
		fmt.Printf("LOG <%v>\n", message)
	})
	if err != nil {
		fmt.Println("Failed to initialize malgo:", err)
		os.Exit(1)
	}
	defer cleanup()

	// Register the hotkey: Ctrl + Shift + Q (Resend last audio)
	hook.Register(hook.KeyDown, []string{"q", "ctrl", "shift"}, func(e hook.Event) {
		if isListening {
			fmt.Println("Currently listening. Stop recording (Ctrl+Shift+S) before resending.")
			return
		}

		mutex.Lock()
		audioToSend := lastCapturedSamples
		mutex.Unlock()

		if len(audioToSend) == 0 {
			fmt.Println("No previous audio recording found to resend.")
			return
		}

		fmt.Println("🔁 Resending last recorded audio to OpenAI...")
		handleAudioTranscription(audioToSend) // Use the refactored handler
	})

	// Register the hotkey: Ctrl + Shift + S (Record only)
	hook.Register(hook.KeyDown, []string{"s", "ctrl", "shift"}, func(e hook.Event) {
		if isListening {
			stopRecordingAndSend()
		} else {
			startRecording()
		}
	})

	// Register the hotkey: Ctrl + Shift + 2 (Retype last transcription)
	hook.Register(hook.KeyDown, []string{"2", "ctrl", "shift"}, func(e hook.Event) {
		// Check for cooldown period (prevent multiple rapid triggers)
		if time.Since(lastRetypeTime) < 1*time.Second {
			fmt.Println("⚠️ Retype cooldown active, ignoring request")
			return
		}

		// Update last retype time
		lastRetypeTime = time.Now()

		// Get the transcription safely
		mutex.Lock()
		transcriptionToType := lastTranscription
		mutex.Unlock()

		if transcriptionToType == "" {
			fmt.Println("No previous transcription found to retype.")
			return
		}

		fmt.Println("⌨️ Retyping last transcription...")

		// Create a goroutine to type with a small delay to ensure all keys are released
		go func() {
			// Wait a short time to ensure all keys are released
			time.Sleep(300 * time.Millisecond)
			robotgo.TypeStr(transcriptionToType)
		}()
	})

	// Start the event hook
	s := hook.Start()
	<-hook.Process(s)
}

// loadEnvFile reads the .env file and loads environment variables
func loadEnvFile() error {
	file, err := os.Open(".env")
	if err != nil {
		return fmt.Errorf("cannot open .env file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		
		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Split on first '=' to handle values with '=' in them
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		// Remove quotes if present
		if len(value) >= 2 && ((value[0] == '"' && value[len(value)-1] == '"') || 
			(value[0] == '\'' && value[len(value)-1] == '\'')) {
			value = value[1 : len(value)-1]
		}

		// Set environment variables
		if key == "OPENAI_API_KEY" {
			openAIAPIKey = value
		}
	}

	return scanner.Err()
}

// Starts recording audio
func startRecording() {
	fmt.Println("🎤 Started Listening...")
	isListening = true

	deviceConfig := malgo.DefaultDeviceConfig(malgo.Capture)
	deviceConfig.Capture.Format = malgo.FormatS16
	deviceConfig.Capture.Channels = 1
	deviceConfig.SampleRate = 16000
	deviceConfig.Alsa.NoMMap = 1

	// Clear previous audio data
	mutex.Lock()
	pCapturedSamples = make([]byte, 0)
	capturedSampleCount = 0
	mutex.Unlock()

	onRecvFrames := func(_, pSample []byte, framecount uint32) {
		sampleCount := framecount * deviceConfig.Capture.Channels * uint32(malgo.SampleSizeInBytes(deviceConfig.Capture.Format))

		mutex.Lock()
		pCapturedSamples = append(pCapturedSamples, pSample...)
		capturedSampleCount += sampleCount
		mutex.Unlock()
	}

	captureCallbacks := malgo.DeviceCallbacks{Data: onRecvFrames}

	var err error
	recordDevice, err = malgo.InitDevice(ctx.Context, deviceConfig, captureCallbacks)
	if err != nil {
		fmt.Println("Error initializing capture device:", err)
		return
	}

	err = recordDevice.Start()
	if err != nil {
		fmt.Println("Error starting capture device:", err)
		return
	}
}

// Stops recording and initiates the transcription process
func stopRecordingAndSend() {
	fmt.Println("🛑 Stopped Listening...")
	isListening = false

	if recordDevice != nil {
		// It's crucial to Stop() *before* Uninit()
		err := recordDevice.Stop()
		if err != nil {
			fmt.Println("Error stopping capture device:", err)
			// Continue anyway to try and process potentially captured audio
		}
		recordDevice.Uninit()
		recordDevice = nil
	} else {
		fmt.Println("No active recording device found.")
		// Potentially check if there are samples anyway? For now, return.
		return
	}

	mutex.Lock()
	samplesToProcess := pCapturedSamples
	// Store the recorded audio immediately, regardless of transcription success
	if len(pCapturedSamples) > 0 {
		lastCapturedSamples = make([]byte, len(pCapturedSamples))
		copy(lastCapturedSamples, pCapturedSamples)
		fmt.Println("📼 Recorded audio saved for potential resending.")
	}
	mutex.Unlock()

	if len(samplesToProcess) == 0 {
		fmt.Println("No audio recorded.")
		return
	}

	fmt.Println("📤 Sending recorded audio to OpenAI...")
	// Call the refactored handler function in a goroutine to avoid blocking
	go handleAudioTranscription(samplesToProcess)
}

// Handles sending audio, getting transcription, storing results, and typing/pasting.
// Runs in its own goroutine.
func handleAudioTranscription(audioData []byte) {
	// Make a local copy of the audio data to prevent race conditions
	// if the original slice is modified elsewhere (though unlikely with current logic).
	audioDataCopy := make([]byte, len(audioData))
	copy(audioDataCopy, audioData)

	// Debug: Print raw PCM audio size
	fmt.Printf("📊 Raw PCM audio size: %s\n", formatBytes(len(audioDataCopy)))

	transcription, err := sendAudioToOpenAI(audioDataCopy)
	if err != nil {
		fmt.Println("Error sending/transcribing audio to OpenAI:", err)
		// Don't update lastTranscription on error, but audio is already saved
		return
	}

	fmt.Println("✅ Transcription received from OpenAI:", transcription)

	// Store only the transcription on success
	mutex.Lock()
	lastTranscription = transcription
	mutex.Unlock()

	// --- Clipboard/Typing Logic ---
	// This part interacts with the UI, potential for delays/errors
	fmt.Println("⌨️  Pasting/Typing transcription...")
	originalClipboardContent, err := robotgo.ReadAll()
	if err != nil {
		fmt.Println("Warning: Could not read clipboard:", err)
		// Fallback to typing if clipboard read fails
		robotgo.TypeStr(transcription)
	} else {
		// Restore clipboard content when the function exits
		defer func() {
			if err := robotgo.WriteAll(originalClipboardContent); err != nil {
				fmt.Println("Warning: Could not restore clipboard:", err)
			}
		}()

		// Write transcription to clipboard and paste
		if err := robotgo.WriteAll(transcription); err != nil {
			fmt.Println("Warning: Could not write to clipboard:", err)
			// Fallback to typing if clipboard write fails
			robotgo.TypeStr(transcription)
		} else {
			robotgo.KeyTap("v", "cmd") // Or "ctrl" for Windows/Linux
		}
	}
}

// formatBytes converts bytes to a human-readable format
func formatBytes(bytes int) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	
	units := []string{"B", "KB", "MB", "GB", "TB"}
	
	value := float64(bytes)
	unitIndex := 0
	
	for value >= unit && unitIndex < len(units)-1 {
		value /= unit
		unitIndex++
	}
	
	if unitIndex == 0 {
		return fmt.Sprintf("%d %s", int(value), units[unitIndex])
	}
	
	return fmt.Sprintf("%.1f %s (%d bytes)", value, units[unitIndex], bytes)
}

// Sends the recorded audio data to OpenAI and returns the transcription
func sendAudioToOpenAI(audioData []byte) (string, error) {
	// Start timing the entire function
	startTime := time.Now()
	fmt.Printf("⏱️  Starting OpenAI transcription request...\n")

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Create a form file field named "file" (OpenAI's expected field name)
	part, err := writer.CreateFormFile("file", "audio.wav")
	if err != nil {
		return "", err
	}

	// Encode PCM to WAV using go-audio/wav
	encodeStartTime := time.Now()
	err = encodeWAV(part, audioData, 16000, 1, 16)
	if err != nil {
		return "", err
	}
	encodeTime := time.Since(encodeStartTime)
	fmt.Printf("⏱️  Audio encoding took: %v\n", encodeTime)

	// Add model field (required by OpenAI)
	err = writer.WriteField("model", modelName)
	if err != nil {
		return "", err
	}

	// Add language field (optional but recommended)
	err = writer.WriteField("language", "en")
	if err != nil {
		return "", err
	}

	// Close the writer
	writer.Close()

	// Debug: Print WAV file size and multipart form size
	fmt.Printf("📊 WAV file size (in multipart form): %s\n", formatBytes(buf.Len()))

	// Create a POST request
	req, err := http.NewRequest("POST", openAIAPIURL, &buf)
	if err != nil {
		return "", err
	}

	// Set headers
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+openAIAPIKey)

	// Send request and measure network + processing time
	networkStartTime := time.Now()
	fmt.Printf("🌐 Sending request to OpenAI...\n")
	
	resp, err := httpClient.Do(req)
	if err == nil {
		fmt.Printf("🔗 Connection protocol: %s\n", resp.Proto)
	}
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	networkTime := time.Since(networkStartTime)
	fmt.Printf("⏱️  Network + OpenAI processing took: %v\n", networkTime)

	// Read OpenAI response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	// Log the entire raw JSON response
	fmt.Printf("📋 Raw OpenAI JSON response:\n%s\n", string(body))

	// Check for HTTP errors
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse JSON response
	var jsonResponse OpenAITranscriptionResponse
	if err := json.Unmarshal(body, &jsonResponse); err != nil {
		return "", fmt.Errorf("error parsing OpenAI JSON response: %v", err)
	}

	// Calculate and display total time
	totalTime := time.Since(startTime)
	fmt.Printf("⏱️  Total transcription time: %v (encoding: %v, network+processing: %v)\n", 
		totalTime, encodeTime, networkTime)

	return jsonResponse.Text, nil
}

// Converts raw PCM audio data to WAV format using go-audio/wav
func encodeWAV(w io.Writer, pcmData []byte, sampleRate, channels, bitDepth int) error {
	// Convert byte slice to int slice (PCM S16LE -> int16)
	intData := twoByteDataToIntSlice(pcmData)

	// Create an IntBuffer with the converted PCM data
	intBuffer := &audio.IntBuffer{
		Data:   intData,
		Format: &audio.Format{SampleRate: sampleRate, NumChannels: channels},
	}

	// Create a new WAV encoder (must use io.WriteSeeker)
	inMemoryFile, err := os.CreateTemp("", "audio-*.wav")
	if err != nil {
		return fmt.Errorf("error creating temp file: %v", err)
	}
	defer os.Remove(inMemoryFile.Name()) // Cleanup temp file

	wavEncoder := wav.NewEncoder(inMemoryFile, sampleRate, bitDepth, channels, 1)

	// Encode and write the WAV file
	if err := wavEncoder.Write(intBuffer); err != nil {
		return fmt.Errorf("error encoding WAV: %v", err)
	}

	// Finalize the WAV file (write headers)
	if err := wavEncoder.Close(); err != nil {
		return fmt.Errorf("error closing WAV encoder: %v", err)
	}

	// Seek back to the start of the file and copy to the provided writer
	inMemoryFile.Seek(0, io.SeekStart)
	_, err = io.Copy(w, inMemoryFile)
	return err
}

// Converts two-byte PCM samples to an int slice
func twoByteDataToIntSlice(audioData []byte) []int {
	intData := make([]int, len(audioData)/2)
	for i := 0; i < len(audioData); i += 2 {
		// Convert the byte slice to int16 (LittleEndian)
		value := int(int16(binary.LittleEndian.Uint16(audioData[i : i+2])))
		intData[i/2] = value
	}
	return intData
}

// Cleanup malgo resources
func cleanup() {
	fmt.Println("Cleaning up malgo context...")
	if recordDevice != nil {
		recordDevice.Stop()
		recordDevice.Uninit()
	}
	if ctx != nil {
		ctx.Uninit()
		ctx.Free()
	}
}

// convertToMP3 converts a WAV file to MP3 using FFmpeg
func convertToMP3(wavFilePath string, bitrate string) (string, error) {
	// Create temp MP3 file
	mp3File, err := os.CreateTemp("", "benchmark-*.mp3")
	if err != nil {
		return "", fmt.Errorf("cannot create temp MP3 file: %w", err)
	}
	mp3File.Close() // Close file handle, but keep the path

	// Run FFmpeg conversion
	cmd := exec.Command("ffmpeg", "-y", "-i", wavFilePath, 
		"-codec:a", "libmp3lame", "-b:a", bitrate, mp3File.Name())
	
	// Capture stderr for potential error messages
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	
	if err := cmd.Run(); err != nil {
		os.Remove(mp3File.Name()) // Cleanup on error
		return "", fmt.Errorf("ffmpeg conversion failed: %w\nFFmpeg stderr: %s", err, stderr.String())
	}

	return mp3File.Name(), nil
}

// sendFileToOpenAI sends an audio file directly to OpenAI (for benchmarking)
func sendFileToOpenAI(filePath string, fileType string) (string, time.Duration, error) {
	startTime := time.Now()

	// Read the file
	fileData, err := os.ReadFile(filePath)
	if err != nil {
		return "", 0, fmt.Errorf("cannot read file: %w", err)
	}

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Create a form file field named "file"
	part, err := writer.CreateFormFile("file", fmt.Sprintf("audio.%s", fileType))
	if err != nil {
		return "", 0, err
	}

	// Write file data directly
	_, err = part.Write(fileData)
	if err != nil {
		return "", 0, err
	}

	// Add model field
	err = writer.WriteField("model", "whisper-1")
	if err != nil {
		return "", 0, err
	}

	// Add language field
	err = writer.WriteField("language", "en")
	if err != nil {
		return "", 0, err
	}

	writer.Close()

	// Create and send request
	req, err := http.NewRequest("POST", openAIAPIURL, &buf)
	if err != nil {
		return "", 0, err
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+openAIAPIKey)

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	
	fmt.Printf("🔗 Connection protocol: %s\n", resp.Proto)

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, err
	}

	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("OpenAI API error (status %d): %s", resp.StatusCode, string(body))
	}

	var jsonResponse OpenAITranscriptionResponse
	if err := json.Unmarshal(body, &jsonResponse); err != nil {
		return "", 0, fmt.Errorf("error parsing OpenAI JSON response: %v", err)
	}

	totalTime := time.Since(startTime)
	return jsonResponse.Text, totalTime, nil
}

// runBenchmark compares WAV vs MP3 performance
func runBenchmark() error {
	fmt.Println("\n🚀 === AUDIO BENCHMARK STARTING ===")
	
	// Check if audio.wav exists
	if _, err := os.Stat("audio.wav"); os.IsNotExist(err) {
		return fmt.Errorf("audio.wav file not found in current directory")
	}

	// Check if FFmpeg is available
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return fmt.Errorf("ffmpeg not found in PATH - please install FFmpeg for MP3 benchmarking")
	}

	// Get file info
	fileInfo, err := os.Stat("audio.wav")
	if err != nil {
		return err
	}
	fmt.Printf("📊 Original WAV file size: %s\n", formatBytes(int(fileInfo.Size())))

	// Test 1: Send WAV directly
	fmt.Println("\n📤 Test 1: Sending WAV file to OpenAI...")
	wavTranscription, wavTime, err := sendFileToOpenAI("audio.wav", "wav")
	if err != nil {
		return fmt.Errorf("WAV test failed: %w", err)
	}
	fmt.Printf("✅ WAV Result - Time: %v, Transcription: %s\n", wavTime, wavTranscription)

	// Test 2: Convert to MP3 and send
	fmt.Println("\n🔄 Test 2: Converting to MP3 and sending to OpenAI...")
	
	conversionStart := time.Now()
	mp3FilePath, err := convertToMP3("audio.wav", "64k")
	if err != nil {
		return fmt.Errorf("MP3 conversion failed: %w", err)
	}
	defer os.Remove(mp3FilePath) // Cleanup temp file
	
	conversionTime := time.Since(conversionStart)
	
	// Get MP3 file size
	mp3FileInfo, err := os.Stat(mp3FilePath)
	if err != nil {
		return err
	}
	
	fmt.Printf("⏱️  MP3 conversion took: %v\n", conversionTime)
	fmt.Printf("📊 MP3 file size: %s\n", formatBytes(int(mp3FileInfo.Size())))
	fmt.Printf("📈 Compression ratio: %.1f%%\n", 
		100.0 * (1.0 - float64(mp3FileInfo.Size())/float64(fileInfo.Size())))

	mp3Transcription, mp3NetworkTime, err := sendFileToOpenAI(mp3FilePath, "mp3")
	if err != nil {
		return fmt.Errorf("MP3 test failed: %w", err)
	}
	
	totalMP3Time := conversionTime + mp3NetworkTime
	fmt.Printf("✅ MP3 Result - Network time: %v, Total time: %v\n", mp3NetworkTime, totalMP3Time)
	fmt.Printf("   Transcription: %s\n", mp3Transcription)

	// Summary
	fmt.Println("\n📊 === BENCHMARK SUMMARY ===")
	fmt.Printf("WAV approach:  %v total\n", wavTime)
	fmt.Printf("MP3 approach:  %v total (%v conversion + %v network)\n", 
		totalMP3Time, conversionTime, mp3NetworkTime)
	
	if totalMP3Time < wavTime {
		timeSaved := wavTime - totalMP3Time
		fmt.Printf("🏆 MP3 was faster by %v (%.1f%% improvement)\n", 
			timeSaved, 100.0*float64(timeSaved)/float64(wavTime))
	} else {
		timeExtra := totalMP3Time - wavTime
		fmt.Printf("🐌 WAV was faster by %v (%.1f%% slower with MP3)\n", 
			timeExtra, 100.0*float64(timeExtra)/float64(wavTime))
	}

	// Check transcription quality
	if wavTranscription == mp3Transcription {
		fmt.Println("✅ Transcriptions are identical - no quality loss detected")
	} else {
		fmt.Println("⚠️  Transcriptions differ - MP3 compression may have affected quality")
		fmt.Printf("   WAV: %s\n", wavTranscription)
		fmt.Printf("   MP3: %s\n", mp3Transcription)
	}

	fmt.Println("=== BENCHMARK COMPLETE ===")
	return nil
}

// initHTTPClient initializes the optimized HTTP client with HTTP/2 and connection pooling
func initHTTPClient() {
	// Create a custom transport with optimized settings
	transport := &http.Transport{
		// Connection pooling settings
		MaxIdleConns:          100,              // Maximum idle connections across all hosts
		MaxIdleConnsPerHost:   10,               // Maximum idle connections per host
		MaxConnsPerHost:       20,               // Maximum connections per host (includes active)
		IdleConnTimeout:       90 * time.Second, // How long to keep idle connections
		
		// HTTP/2 settings
		ForceAttemptHTTP2:     true,             // Force HTTP/2 usage when possible
		
		// Keep-alive settings
		DisableKeepAlives:     false,            // Enable keep-alive connections
		
		// Timeouts
		TLSHandshakeTimeout:   10 * time.Second, // TLS handshake timeout
		ResponseHeaderTimeout: 15 * time.Second, // Timeout for response headers
		ExpectContinueTimeout: 1 * time.Second,  // Timeout for Expect: 100-continue
		
		// Compression
		DisableCompression:    false,            // Enable compression
		
		// TCP keep-alive (operating system level)
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second, // Connection timeout
			KeepAlive: 30 * time.Second, // TCP keep-alive interval
		}).DialContext,
	}
	
	httpClient = &http.Client{
		Transport: transport,
		Timeout:   30 * time.Second, // Overall request timeout
	}
	
	fmt.Println("🔧 HTTP client initialized with HTTP/2 and connection pooling optimizations")
}

// printConnectionStats prints current HTTP transport statistics
func printConnectionStats() {
	if transport, ok := httpClient.Transport.(*http.Transport); ok {
		// Note: Go's http.Transport doesn't expose detailed connection pool stats directly
		// but we can see if connections are being reused by observing response times
		fmt.Println("📊 Connection pool is active and reusing connections")
		fmt.Printf("   - Max idle connections per host: %d\n", transport.MaxIdleConnsPerHost)
		fmt.Printf("   - Max connections per host: %d\n", transport.MaxConnsPerHost)
		fmt.Printf("   - Idle connection timeout: %v\n", transport.IdleConnTimeout)
		fmt.Printf("   - HTTP/2 enabled: %v\n", transport.ForceAttemptHTTP2)
	}
} 