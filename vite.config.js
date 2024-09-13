import { defineConfig } from 'vite';
import path from 'path';
import { copyFileSync } from 'fs';

export default defineConfig({
  base: '/typing/bigram-prolific-study/',
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
        copyFileSync('bigram_tables/bigram_3pairs_LH.csv', 'dist/bigram_tables/bigram_3pairs_LH.csv');
        copyFileSync('bigram_tables/bigram_3pairs_RH.csv', 'dist/bigram_tables/bigram_3pairs_RH.csv');
        copyFileSync('bigram_tables/bigram_80pairs_LH.csv', 'dist/bigram_tables/bigram_80pairs_LH.csv');
        copyFileSync('bigram_tables/bigram_80pairs_RH.csv', 'dist/bigram_tables/bigram_80pairs_RH.csv');
        copyFileSync('bigram_tables/bigram_2x80pairs_LH.csv', 'dist/bigram_tables/bigram_2x80pairs_LH.csv');
        copyFileSync('bigram_tables/bigram_2x80pairs_RH.csv', 'dist/bigram_tables/bigram_2x80pairs_RH.csv');
      }
    }
  ]
});

