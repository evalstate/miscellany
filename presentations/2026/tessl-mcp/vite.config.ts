import { defineConfig, type Plugin } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

function disableManualChunksForSingleFile(): Plugin {
  return {
    name: 'disable-manual-chunks-for-single-file',
    enforce: 'post',
    configResolved(config) {
      if (process.env.SLIDEV_SINGLEFILE !== '1') return

      const ro = config.build.rollupOptions ||= {}
      const output: any = ro.output ||= {}

      if (Array.isArray(output)) {
        for (const item of output) {
          item.manualChunks = undefined
          item.inlineDynamicImports = undefined
          item.codeSplitting = false
        }
      } else {
        output.manualChunks = undefined
        output.inlineDynamicImports = undefined
        output.codeSplitting = false
      }
    },
  }
}

// Normal Slidev dev/build behaves normally.
// Set SLIDEV_SINGLEFILE=1 to produce a best-effort filesystem-viewable single HTML file.
export default defineConfig(() => {
  const single = process.env.SLIDEV_SINGLEFILE === '1'

  return {
    plugins: single
      ? [viteSingleFile(), disableManualChunksForSingleFile()]
      : [],
    build: single
      ? {
          assetsInlineLimit: Number.MAX_SAFE_INTEGER,
          cssCodeSplit: false,
          rollupOptions: {
            output: {
              codeSplitting: false,
              manualChunks: undefined,
              inlineDynamicImports: undefined,
            },
          },
        }
      : {},
  }
})
