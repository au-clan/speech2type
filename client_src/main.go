package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"sync"

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
	serverURL           = "http://localhost:8000/speechtotext"
)

func main() {
	fmt.Println("--- Press Ctrl + Shift + S to start/stop listening ---")

	// Initialize malgo
	var err error
	ctx, err = malgo.InitContext(nil, malgo.ContextConfig{}, func(message string) {
		fmt.Printf("LOG <%v>\n", message)
	})
	if err != nil {
		fmt.Println("Failed to initialize malgo:", err)
		os.Exit(1)
	}
	defer cleanup()

	// Register the hotkey: Ctrl + Shift + S
	hook.Register(hook.KeyDown, []string{"s", "ctrl", "shift"}, func(e hook.Event) {
		if isListening {
			stopRecordingAndSend()
		} else {
			startRecording()
		}
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

// Stops recording, sends audio, and types the transcription
func stopRecordingAndSend() {
	fmt.Println("🛑 Stopped Listening...")
	isListening = false

	if recordDevice != nil {
		recordDevice.Stop()
		recordDevice.Uninit()
		recordDevice = nil
	}

	if len(pCapturedSamples) == 0 {
		fmt.Println("No audio recorded.")
		return
	}

	fmt.Println("📤 Sending recorded audio to server...")
	transcription, err := sendAudioToServer(pCapturedSamples)
	if err != nil {
		fmt.Println("Error sending audio:", err)
	} else {
		fmt.Println("✅ Transcription received:", transcription)

		// Type the transcribed text using RobotGo
		fmt.Println("⌨️  Typing transcription...")
		robotgo.TypeStr(transcription)
	}
}

// Sends the recorded audio data to the server and returns the transcription
func sendAudioToServer(audioData []byte) (string, error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Create a form file field named "audio_file"
	part, err := writer.CreateFormFile("audio_file", "audio.wav")
	if err != nil {
		return "", err
	}

	// Encode PCM to WAV using go-audio/wav
	err = encodeWAV(part, audioData, 16000, 1, 16)
	if err != nil {
		return "", err
	}

	// Close the writer
	writer.Close()

	// Create a POST request
	req, err := http.NewRequest("POST", serverURL, &buf)
	if err != nil {
		return "", err
	}

	// Set headers
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send request
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// Read server response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	// Parse JSON response
	var jsonResponse TranscriptionResponse
	if err := json.Unmarshal(body, &jsonResponse); err != nil {
		return "", fmt.Errorf("error parsing JSON response: %v", err)
	}

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
}
