import { useRef, useCallback, useState, useEffect } from "react";
import { Box, Button, Typography, CircularProgress } from "@mui/material";

interface KetcherEditorProps {
  onSmilesChange?: (smiles: string) => void;
  darkMode?: boolean;
}

// Safely access the ketcher API from the iframe's contentWindow
function getKetcher(iframe: HTMLIFrameElement | null) {
  try {
    return (iframe?.contentWindow as any)?.ketcher ?? null;
  } catch {
    return null;
  }
}

export default function KetcherEditor({ onSmilesChange, darkMode = false }: KetcherEditorProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [ready, setReady] = useState(false);
  const [retryKey, setRetryKey] = useState(0);

  // After the iframe's HTML loads, poll until the ketcher WASM object appears
  const handleLoad = useCallback(() => {
    let attempts = 0;
    const MAX = 80; // 40 s max (500 ms × 80)
    const poll = () => {
      const k = getKetcher(iframeRef.current);
      if (k) {
        setReady(true);
      } else if (attempts < MAX) {
        attempts++;
        setTimeout(poll, 500);
      } else {
        // Force-reload by bumping the iframe key
        setRetryKey((n) => n + 1);
      }
    };
    poll();
  }, []);

  // Reset ready state whenever we force-reload the iframe
  useEffect(() => {
    setReady(false);
  }, [retryKey]);

  // Apply theme to Ketcher whenever ready or darkMode changes
  useEffect(() => {
    if (!ready) return;
    const k = getKetcher(iframeRef.current);
    if (!k) return;
    try {
      k.setSettings({ theme: darkMode ? 'dark' : 'default' });
    } catch (e) {
      console.warn('Ketcher setSettings(theme) failed:', e);
    }
  }, [ready, darkMode]);

  const handleGetSmiles = useCallback(async () => {
    const k = getKetcher(iframeRef.current);
    if (!k) return;
    try {
      const smiles: string = await k.getSmiles();
      if (smiles) onSmilesChange?.(smiles);
    } catch (e) {
      console.error("Failed to get SMILES from Ketcher:", e);
    }
  }, [onSmilesChange]);

  return (
    <Box sx={{ width: "100%" }}>
      {/* Spinner shown while Ketcher WASM initialises */}
      {!ready && (
        <Box
          sx={{
            height: 520,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 2,
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 1,
          }}
        >
          <CircularProgress size={36} />
          <Typography variant="body2" color="text.secondary">
            Loading Ketcher editor…
          </Typography>
        </Box>
      )}

      {/* Iframe is always mounted so Ketcher loads in the background */}
      <iframe
        key={retryKey}
        ref={iframeRef}
        src="/ketcher/standalone/index.html"
        title="Ketcher molecule editor"
        onLoad={handleLoad}
        style={{
          width: "100%",
          height: 520,
          border: "1px solid #c0c0c0",
          borderRadius: 4,
          display: ready ? "block" : "none",
        }}
      />

      <Box sx={{ mt: 1, display: "flex", alignItems: "center", gap: 1.5 }}>
        <Button
          variant="outlined"
          size="small"
          disabled={!ready}
          onClick={handleGetSmiles}
        >
          Use Structure
        </Button>
        <Typography variant="caption" color="text.secondary">
          Draw a molecule above, then click{" "}
          <strong>Use Structure</strong> to populate the SMILES input.
        </Typography>
      </Box>
    </Box>
  );
}
