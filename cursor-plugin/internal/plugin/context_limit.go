package plugin

import (
	"encoding/json"

	"cursorplugin/internal/openai"
)

const (
	clientContextLimit int64 = 1_000_000
	contextReserve     int64 = 16_384
	contextCountMethod       = "conservative_utf8_bytes_v1"
)

type contextLimitError struct{ reason string }

func (err *contextLimitError) Error() string {
	return "Cursor client context policy (1000000 conservative budget units): " + err.reason
}

func (err *contextLimitError) Is(target error) bool { return target == openai.ErrInvalidRequest }

func contextExceeded(reason string) error { return &contextLimitError{reason: reason} }

type contextBudgetWire struct {
	Messages            []json.RawMessage `json:"messages"`
	Tools               []json.RawMessage `json:"tools"`
	MaxTokens           int64             `json:"max_tokens"`
	MaxCompletionTokens int64             `json:"max_completion_tokens"`
}

func decodeContextRequest(raw []byte) (executorRequest, openai.ChatRequest, int64, error) {
	var request executorRequest
	if err := json.Unmarshal(raw, &request); err != nil {
		return request, openai.ChatRequest{}, 0, &openai.InvalidRequestError{Message: "decode executor request: " + err.Error()}
	}
	payload := request.Payload
	if len(payload) == 0 {
		payload = request.OriginalRequest
	}
	var wire contextBudgetWire
	if err := json.Unmarshal(payload, &wire); err != nil {
		return request, openai.ChatRequest{}, 0, &openai.InvalidRequestError{Message: "decode context budget: " + err.Error()}
	}
	if wire.MaxTokens < 0 || wire.MaxCompletionTokens < 0 {
		return request, openai.ChatRequest{}, 0, &openai.InvalidRequestError{Message: "output token limits must be nonnegative integers"}
	}
	chat, err := openai.ParseChatRequest(payload)
	if err != nil {
		return request, openai.ChatRequest{}, 0, err
	}
	// No client-side admission budget: the 1-byte-per-unit heuristic rejected
	// requests well below real model capacity, so Cursor is now the only
	// authority on what fits. The estimate below is informational (token counts)
	// and never rejects a request. No local output truncation is applied either.
	tokens := max(int64(len(payload)), contextInputBytes(chat))
	return request, chat, tokens, nil
}

// contextInputBytes is an informational size estimate used only for token
// counts. It is not an admission budget and never rejects a request.
func contextInputBytes(chat openai.ChatRequest) int64 {
	units := int64(len(chat.System)) + int64(len(chat.Prompt)) + int64(len(chat.Transcript))*32
	for _, tool := range chat.Tools {
		units += int64(len(tool.Name)) + int64(len(tool.Description)) + int64(len(tool.Parameters)) + 256
	}
	for _, attachment := range chat.Attachments {
		units += int64(len(attachment.Name)) + int64(len(attachment.Content)) + 256
	}
	for _, image := range chat.Images {
		units += int64(len(image.Name)) + int64(len(image.MIMEType)) + max(int64(len(image.Data)), 65_536)
	}
	return units
}

func effectiveContextLength(native int64) int64 {
	if native <= 0 {
		return 0
	}
	return min(native, clientContextLimit)
}
