package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/gen2brain/malgo"
	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/go-vgo/robotgo"
	hook "github.com/robotn/gohook"
)

type TranscriptionResponse struct {
	Transcription string `json:"transcription"`
}

var (
	isListening         bool
	ctx                 *malgo.AllocatedContext
	recordDevice        *malgo.Device
	pCapturedSamples    []byte
	capturedSampleCount uint32
	mutex               sync.Mutex
	serverURL           = "http://localhost:8123/speechtotext"

	// Store the last successfully recorded audio and its transcription
	lastCapturedSamples []byte
	lastTranscription   string

	// Add cooldown tracking to prevent rapid repeated triggers
	lastRetypeTime time.Time
	
	// Optimized HTTP client for connection reuse and HTTP/2
	httpClient *http.Client
	
	// SSH tunnel process
	sshTunnelCmd *exec.Cmd
)

// startSSHTunnel starts the autossh tunnel in the background
func startSSHTunnel() error {
	// Check if autossh is available
	if _, err := exec.LookPath("autossh"); err != nil {
		return fmt.Errorf("autossh not found in PATH - please install autossh for automatic SSH tunneling")
	}

	fmt.Println("🔗 Setting up SSH tunnel to spacedexposure...")
	
	// Command: autossh -M 0 -N -L 8123:localhost:8123 spacedexposure
	sshTunnelCmd = exec.Command("autossh", "-M", "0", "-N", "-L", "8123:localhost:8123", "spacedexposure")
	
	// Start the command in the background
	err := sshTunnelCmd.Start()
	if err != nil {
		return fmt.Errorf("failed to start SSH tunnel: %v", err)
	}
	
	fmt.Printf("✅ SSH tunnel started (PID: %d) - forwarding localhost:8123 to spacedexposure:8123\n", sshTunnelCmd.Process.Pid)
	
	// Start a goroutine to monitor the process
	go func() {
		err := sshTunnelCmd.Wait()
		if err != nil {
			fmt.Printf("⚠️  SSH tunnel process exited with error: %v\n", err)
		} else {
			fmt.Println("ℹ️  SSH tunnel process exited normally")
		}
	}()
	
	// Give the tunnel a moment to establish
	time.Sleep(2 * time.Second)
	fmt.Println("🌉 SSH tunnel should now be ready for connections")
	
	return nil
}

func main() {
	fmt.Println("🚀 Speech2Type - Localhost Server Mode")
	fmt.Println("--- Press Ctrl + Shift + S to start/stop listening ---")
	fmt.Println("--- Press Ctrl + Shift + Q to resend last audio ---")
	fmt.Println("--- Press Ctrl + Shift + 2 to retype last transcription ---")

	// Initialize optimized HTTP client
	initHTTPClient()
	var err error;

	// Start SSH tunnel
	err = startSSHTunnel()
	if err != nil {
		fmt.Printf("⚠️  SSH tunnel setup failed: %v\n", err)
		fmt.Println("Continuing without tunnel - make sure your server is accessible on localhost:8123")
	}

	// Run benchmark if audio.wav file exists
	// err := runBenchmark()
	// if err != nil {
	// 	fmt.Printf("⚠️  Benchmark failed: %v\n", err)
	// 	fmt.Println("Continuing with normal operation...")
	// }

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

		fmt.Println("🔁 Resending last recorded audio...")
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

	fmt.Println("📤 Sending recorded audio to server...")
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

	transcription, err := sendAudioToServer(audioDataCopy)
	if err != nil {
		fmt.Println("Error sending/transcribing audio:", err)
		// Don't update lastTranscription on error, but audio is already saved
		return
	}

	fmt.Println("✅ Transcription received:", transcription)

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

// Sends the recorded audio data to the server and returns the transcription
func sendAudioToServer(audioData []byte) (string, error) {
	// Start timing the entire function
	startTime := time.Now()
	fmt.Printf("⏱️  Starting localhost transcription request...\n")

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Create a form file field named "audio_file"
	part, err := writer.CreateFormFile("audio_file", "audio.wav")
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

	// Close the writer
	writer.Close()

	// Debug: Print WAV file size
	fmt.Printf("📊 WAV file size (in multipart form): %s\n", formatBytes(buf.Len()))

	// Create a POST request
	req, err := http.NewRequest("POST", serverURL, &buf)
	if err != nil {
		return "", err
	}

	// Set headers
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send request and measure network + processing time
	networkStartTime := time.Now()
	fmt.Printf("🌐 Sending request to localhost server...\n")
	
	resp, err := httpClient.Do(req)
	if err == nil {
		fmt.Printf("🔗 Connection protocol: %s\n", resp.Proto)
	}
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	networkTime := time.Since(networkStartTime)
	fmt.Printf("⏱️  Network + localhost processing took: %v\n", networkTime)

	// Read server response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	// Log the entire raw JSON response
	fmt.Printf("📋 Raw localhost JSON response:\n%s\n", string(body))

	// Check for HTTP errors
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("localhost server error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse JSON response
	var jsonResponse TranscriptionResponse
	if err := json.Unmarshal(body, &jsonResponse); err != nil {
		return "", fmt.Errorf("error parsing JSON response: %v", err)
	}

	// Calculate and display total time
	totalTime := time.Since(startTime)
	fmt.Printf("⏱️  Total transcription time: %v (encoding: %v, network+processing: %v)\n", 
		totalTime, encodeTime, networkTime)

	return jsonResponse.Transcription, nil
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
	
	// Terminate SSH tunnel
	if sshTunnelCmd != nil && sshTunnelCmd.Process != nil {
		fmt.Printf("🔗 Terminating SSH tunnel (PID: %d)...\n", sshTunnelCmd.Process.Pid)
		err := sshTunnelCmd.Process.Kill()
		if err != nil {
			fmt.Printf("⚠️  Error terminating SSH tunnel: %v\n", err)
		} else {
			fmt.Println("✅ SSH tunnel terminated")
		}
	}
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

// formatBytes formats a byte count into a human-readable string
func formatBytes(bytes int) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
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

// sendFileToServer sends an audio file directly to localhost server (for benchmarking)
func sendFileToServer(filePath string, fileType string) (string, time.Duration, error) {
	startTime := time.Now()

	// Read the file
	fileData, err := os.ReadFile(filePath)
	if err != nil {
		return "", 0, fmt.Errorf("cannot read file: %w", err)
	}

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Create a form file field named "audio_file" (matching server expectations)
	part, err := writer.CreateFormFile("audio_file", fmt.Sprintf("audio.%s", fileType))
	if err != nil {
		return "", 0, err
	}

	// Write file data directly
	_, err = part.Write(fileData)
	if err != nil {
		return "", 0, err
	}

	writer.Close()

	// Create and send request
	req, err := http.NewRequest("POST", serverURL, &buf)
	if err != nil {
		return "", 0, err
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

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
		return "", 0, fmt.Errorf("localhost server error (status %d): %s", resp.StatusCode, string(body))
	}

	var jsonResponse TranscriptionResponse
	if err := json.Unmarshal(body, &jsonResponse); err != nil {
		return "", 0, fmt.Errorf("error parsing JSON response: %v", err)
	}

	totalTime := time.Since(startTime)
	return jsonResponse.Transcription, totalTime, nil
}

// runBenchmark compares WAV vs MP3 performance against localhost server with multiple iterations
func runBenchmark() error {
	const numIterations = 10
	fmt.Printf("\n🚀 === LOCALHOST AUDIO BENCHMARK STARTING (%d iterations) ===\n", numIterations)
	
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

	// Convert to MP3 once (reuse for all iterations)
	fmt.Println("\n🔄 Pre-converting to MP3 for benchmark...")
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

	// Storage for results
	wavTimes := make([]time.Duration, numIterations)
	mp3Times := make([]time.Duration, numIterations)
	var firstWavTranscription, firstMp3Transcription string

	fmt.Printf("\n📤 Running %d iterations of each test...\n", numIterations)

	// Run WAV tests
	fmt.Println("\n=== WAV TESTS ===")
	for i := 0; i < numIterations; i++ {
		fmt.Printf("WAV Test %d/%d: ", i+1, numIterations)
		transcription, duration, err := sendFileToServer("audio.wav", "wav")
		if err != nil {
			return fmt.Errorf("WAV test %d failed: %w", i+1, err)
		}
		wavTimes[i] = duration
		if i == 0 {
			firstWavTranscription = transcription
		}
		fmt.Printf("%v\n", duration)
	}

	// Run MP3 tests
	fmt.Println("\n=== MP3 TESTS ===")
	for i := 0; i < numIterations; i++ {
		fmt.Printf("MP3 Test %d/%d: ", i+1, numIterations)
		transcription, duration, err := sendFileToServer(mp3FilePath, "mp3")
		if err != nil {
			return fmt.Errorf("MP3 test %d failed: %w", i+1, err)
		}
		mp3Times[i] = duration
		if i == 0 {
			firstMp3Transcription = transcription
		}
		fmt.Printf("%v\n", duration)
	}

	// Calculate statistics
	wavStats := calculateStats(wavTimes)
	mp3Stats := calculateStats(mp3Times)

	// Calculate total MP3 times (including conversion)
	totalMp3Stats := BenchmarkStats{
		Average: mp3Stats.Average + conversionTime,
		Min:     mp3Stats.Min + conversionTime,
		Max:     mp3Stats.Max + conversionTime,
		StdDev:  mp3Stats.StdDev, // StdDev remains the same since conversion time is constant
	}

	// Display results
	fmt.Printf("\n📊 === LOCALHOST BENCHMARK RESULTS (%d iterations) ===\n", numIterations)
	
	fmt.Println("\n🎯 WAV Approach (network only):")
	fmt.Printf("   Average: %v\n", wavStats.Average)
	fmt.Printf("   Min:     %v\n", wavStats.Min)
	fmt.Printf("   Max:     %v\n", wavStats.Max)
	fmt.Printf("   StdDev:  %v\n", wavStats.StdDev)
	
	fmt.Println("\n🎯 MP3 Approach (network only):")
	fmt.Printf("   Average: %v\n", mp3Stats.Average)
	fmt.Printf("   Min:     %v\n", mp3Stats.Min)
	fmt.Printf("   Max:     %v\n", mp3Stats.Max)
	fmt.Printf("   StdDev:  %v\n", mp3Stats.StdDev)
	
	fmt.Printf("\n🎯 MP3 Approach (total with %v conversion):\n", conversionTime)
	fmt.Printf("   Average: %v\n", totalMp3Stats.Average)
	fmt.Printf("   Min:     %v\n", totalMp3Stats.Min)
	fmt.Printf("   Max:     %v\n", totalMp3Stats.Max)
	fmt.Printf("   StdDev:  %v\n", totalMp3Stats.StdDev)

	// Performance comparison
	fmt.Println("\n🏁 === PERFORMANCE COMPARISON ===")
	if totalMp3Stats.Average < wavStats.Average {
		timeSaved := wavStats.Average - totalMp3Stats.Average
		improvement := 100.0 * float64(timeSaved) / float64(wavStats.Average)
		fmt.Printf("🏆 MP3 (total) was faster by %v (%.1f%% improvement)\n", timeSaved, improvement)
	} else {
		timeExtra := totalMp3Stats.Average - wavStats.Average
		slower := 100.0 * float64(timeExtra) / float64(wavStats.Average)
		fmt.Printf("🐌 WAV was faster by %v (%.1f%% slower with MP3)\n", timeExtra, slower)
	}

	if mp3Stats.Average < wavStats.Average {
		timeSaved := wavStats.Average - mp3Stats.Average
		improvement := 100.0 * float64(timeSaved) / float64(wavStats.Average)
		fmt.Printf("⚡ MP3 (network only) was faster by %v (%.1f%% improvement)\n", timeSaved, improvement)
	} else {
		timeExtra := mp3Stats.Average - wavStats.Average
		slower := 100.0 * float64(timeExtra) / float64(wavStats.Average)
		fmt.Printf("📶 WAV (network only) was faster by %v (%.1f%% slower with MP3)\n", timeExtra, slower)
	}

	// Consistency analysis
	fmt.Println("\n📈 === CONSISTENCY ANALYSIS ===")
	wavCV := float64(wavStats.StdDev) / float64(wavStats.Average) * 100
	mp3CV := float64(mp3Stats.StdDev) / float64(mp3Stats.Average) * 100
	fmt.Printf("WAV Coefficient of Variation: %.1f%%\n", wavCV)
	fmt.Printf("MP3 Coefficient of Variation: %.1f%%\n", mp3CV)
	
	if wavCV < mp3CV {
		fmt.Println("✅ WAV times are more consistent")
	} else {
		fmt.Println("✅ MP3 times are more consistent")
	}

	// Check transcription quality
	fmt.Println("\n🔍 === TRANSCRIPTION QUALITY CHECK ===")
	if firstWavTranscription == firstMp3Transcription {
		fmt.Println("✅ Transcriptions are identical - no quality loss detected")
		fmt.Printf("   Transcription: %s\n", firstWavTranscription)
	} else {
		fmt.Println("⚠️  Transcriptions differ - MP3 compression may have affected quality")
		fmt.Printf("   WAV: %s\n", firstWavTranscription)
		fmt.Printf("   MP3: %s\n", firstMp3Transcription)
	}

	fmt.Println("=== LOCALHOST BENCHMARK COMPLETE ===")
	return nil
}

// BenchmarkStats holds statistical data for benchmark results
type BenchmarkStats struct {
	Average time.Duration
	Min     time.Duration
	Max     time.Duration
	StdDev  time.Duration
}

// calculateStats computes statistical measures for a slice of durations
func calculateStats(times []time.Duration) BenchmarkStats {
	if len(times) == 0 {
		return BenchmarkStats{}
	}

	// Calculate average
	var sum time.Duration
	for _, t := range times {
		sum += t
	}
	avg := sum / time.Duration(len(times))

	// Find min and max
	min := times[0]
	max := times[0]
	for _, t := range times {
		if t < min {
			min = t
		}
		if t > max {
			max = t
		}
	}

	// Calculate standard deviation
	var sumSquaredDiffs float64
	for _, t := range times {
		diff := float64(t - avg)
		sumSquaredDiffs += diff * diff
	}
	variance := sumSquaredDiffs / float64(len(times))
	stdDev := time.Duration(math.Sqrt(variance))

	return BenchmarkStats{
		Average: avg,
		Min:     min,
		Max:     max,
		StdDev:  stdDev,
	}
}
