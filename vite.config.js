import { defineConfig } from 'vite';
import path from 'path';
import { copyFileSync } from 'fs';

export default defineConfig({
  base: '/typing/bigram-comfort-study/',
  resolve: {
    alias: {
      'jspsych': path.resolve(__dirname, 'node_modules/jspsych'),
      '@jspsych': path.resolve(__dirname, 'node_modules/@jspsych'),
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
  optimizeDeps: {
    include: ['jspsych']
  },
  plugins: [
    {
      name: 'copy-files',
      writeBundle() {
        //copyFileSync('configs/token.json', 'dist/configs/token.json');
        copyFileSync('bigram_3pairs.csv', 'dist/bigram_3pairs.csv');
        copyFileSync('bigram_80pairs.csv', 'dist/bigram_80pairs.csv');
      }
    }
  ]
});

