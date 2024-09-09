import { defineConfig } from 'vite';

import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: 'dist',
  },
  resolve: {
    alias: {
      'jspsych': 'node_modules/jspsych/dist/index.js',
    },
  },
});