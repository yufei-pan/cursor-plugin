package plugin

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_Context_limit_boundary_removed_passes_all_sizes(t *testing.T) {
	empty := map[string]any{"model": "cursor/auto", "messages": []map[string]any{textMessage("user", "")}}
	encoded, err := json.Marshal(empty)
	require.NoError(t, err)
	for _, extra := range []int{0, 1} {
		body := map[string]any{"model": "cursor/auto", "messages": []map[string]any{textMessage("user", strings.Repeat("a", 1_000_000-16_384-len(encoded)+extra))}}
		raw := contextRequestFixture(t, body, false)
		handler := NewHandler(Dependencies{Cursor: &recordingCursorClient{}})
		response, ok := handler.CallWithStatus(context.Background(), "executor.count_tokens", raw)
		require.True(t, ok, string(response))
	}
	request := executorRequest{Payload: []byte(`{"model":"cursor/auto","messages":[{"role":"user","content":"` + strings.Repeat(`\u0061`, 170_000) + `"}]}`)}
	raw, err := json.Marshal(request)
	require.NoError(t, err)
	handler := NewHandler(Dependencies{Cursor: &recordingCursorClient{}})
	response, ok := handler.CallWithStatus(context.Background(), "executor.count_tokens", raw)
	require.True(t, ok, string(response))
}

func Test_Context_limit_passes_structural_and_media_overhead(t *testing.T) {
	messages := make([]map[string]any, 8193)
	for i := range messages {
		messages[i] = textMessage("user", "a")
	}
	tools := make([]map[string]any, 257)
	for i := range tools {
		tools[i] = map[string]any{"type": "function", "function": map[string]any{"name": "read"}}
	}
	images := make([]map[string]any, 16)
	for i := range images {
		images[i] = map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,aW1hZ2U="}}
	}
	cases := []map[string]any{
		{"model": "cursor/auto", "messages": messages},
		{"model": "cursor/auto", "messages": []map[string]any{textMessage("user", "hello")}, "tools": tools},
		{"model": "cursor/auto", "messages": []map[string]any{{"role": "user", "content": images}}},
		{"model": "cursor/auto", "messages": []map[string]any{textMessage("user", strings.Repeat("a", 500_000))}, "max_tokens": 600_000},
	}
	for _, body := range cases {
		handler := NewHandler(Dependencies{Cursor: &recordingCursorClient{}})
		response, ok := handler.CallWithStatus(context.Background(), "executor.count_tokens", contextRequestFixture(t, body, false))
		require.True(t, ok, string(response))
		var result envelope
		require.NoError(t, json.Unmarshal(response, &result))
		require.True(t, result.OK)
	}
}

func Test_Context_limit_rejects_invalid_output_limits_without_upstream(t *testing.T) {
	for _, value := range []any{-1, "100", 1.5} {
		client := &recordingCursorClient{}
		handler := NewHandler(Dependencies{Cursor: client})
		raw := contextRequestFixture(t, map[string]any{"model": "cursor/auto", "messages": []map[string]any{textMessage("user", "hello")}, "max_tokens": value}, false)
		response, ok := handler.CallWithStatus(context.Background(), "executor.execute", raw)
		require.False(t, ok, string(response))
		require.Empty(t, client.Inputs())
		var result envelope
		require.NoError(t, json.Unmarshal(response, &result))
		require.Equal(t, 400, result.Error.HTTPStatus)
		require.False(t, result.Error.Retryable)
	}
}

func Test_Context_limit_full_history_passes_to_cursor(t *testing.T) {
	client := &recordingCursorClient{steps: []cursorRunStep{successfulTextStep("first", "conversation", []byte("checkpoint")), successfulTextStep("next", "conversation", []byte("checkpoint")), successfulTextStep("third", "conversation", []byte("checkpoint"))}}
	handler := NewHandler(Dependencies{Cursor: client})
	messages := []map[string]any{textMessage("user", "hello")}
	response, ok := handler.CallWithStatus(context.Background(), "executor.execute", executorFixture(t, "session", "account", "auth", "auto", "", messages))
	require.True(t, ok, string(response))
	messages = append(messages, textMessage("assistant", "first"), textMessage("user", strings.Repeat("x", 1_000_001)))
	response, ok = handler.CallWithStatus(context.Background(), "executor.execute", executorFixture(t, "session", "account", "auth", "auto", "", messages))
	require.True(t, ok, string(response))
	require.Len(t, client.Inputs(), 2)
	messages[2] = textMessage("user", "continue")
	response, ok = handler.CallWithStatus(context.Background(), "executor.execute", executorFixture(t, "session", "account", "auth", "auto", "", messages))
	require.True(t, ok, string(response))
	require.Len(t, client.Inputs(), 3)
}
