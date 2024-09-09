import { defineConfig } from 'vite';

export default defineConfig({
  resolve: {
    alias: {
      'jspsych': 'node_modules/jspsych/dist/index.js',
    },
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: 'index.html',
      },
    },
  },
});