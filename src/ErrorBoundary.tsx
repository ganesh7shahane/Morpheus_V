import { Component, type ReactNode } from "react";
import { Alert, Box } from "@mui/material";

interface Props {
  children: ReactNode;
  fallbackMessage?: string;
}

interface State {
  hasError: boolean;
  message: string;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(error: unknown): State {
    const message =
      error instanceof Error ? error.message : String(error);
    return { hasError: true, message };
  }

  componentDidCatch(error: unknown, info: unknown) {
    console.error("ErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box sx={{ p: 2 }}>
          <Alert severity="warning">
            {this.props.fallbackMessage ?? `Component failed to load: ${this.state.message}`}
          </Alert>
          {!this.props.fallbackMessage && (
            <Box sx={{ mt: 1, fontSize: 11, color: 'text.secondary', fontFamily: 'monospace', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
              {this.state.message}
            </Box>
          )}
        </Box>
      );
    }
    return this.props.children;
  }
}
