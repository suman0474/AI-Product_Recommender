import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    proxy: {
      '^/(api|validate|analyze|schema|login|register|logout|user|new-search|additional_requirements|structure_requirements|upload|search_pdfs|get-price-review|feedback|get_field_description|get_all_field_descriptions|vendors|submodel-mapping|admin|standardization)': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  plugins: [
    react(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Remove console.log statements in production builds
  esbuild: {
    drop: mode === 'production' ? ['console', 'debugger'] : [],
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            // Separate heavy libraries into their own chunks
            if (id.includes('jspdf')) {
              return 'vendor-jspdf';
            }
            // Radix UI components in separate chunk
            if (id.includes('@radix-ui')) {
              return 'vendor-radix';
            }
            // React Query in separate chunk
            if (id.includes('@tanstack')) {
              return 'vendor-tanstack';
            }
            // Markdown renderer
            if (id.includes('react-markdown') || id.includes('remark') || id.includes('rehype')) {
              return 'vendor-markdown';
            }
            // Icons
            if (id.includes('lucide-react')) {
              return 'vendor-icons';
            }
            // Put all other small libraries into a general vendor file
            return 'vendor';
          }
        },
      },
    },
    // Increase limit to 1000kb to ensure the warning stays away
    chunkSizeWarningLimit: 1000,
  },
}));