// import "@/styles/globals.css";
import type { AppProps } from "next/app";

// src/pages/_app.tsx
import 'bootstrap/dist/css/bootstrap.min.css'; // Importação do Bootstrap
import '../styles/globals.css'; // Se você tiver outros estilos globais


export default function App({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />;
}
