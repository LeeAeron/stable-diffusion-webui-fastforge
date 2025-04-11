import { type UserConfig, type Plugin } from 'vite'
import { type OutputOptions } from 'rollup'
import { resolve } from 'path'
import { rename } from 'fs/promises'

export default function (): Plugin {
    return {
        name: 'extension',
        async writeBundle(options, bundle) {
            for (const key in bundle) {
                const b = bundle[key]
                const type = b.type
                if (type !== 'chunk' && type !== 'asset') {
                    continue
                }
                const fileName = resolve(options.dir, b.fileName)
                await rename(fileName, fileName.replace(/\?[0-9a-f]+$/, ''))
            }
        },
        config(config, env) {
            const common: UserConfig = {
                base: './',
                resolve: {
                    alias: [
                        {
                            find: /.*\/assets(\.ts)?$/,
                            replacement: resolve(
                                __dirname,
                                'src/environments/extension/assets.ts'
                            ),
                        },
                    ],
                },
                build: {
                    target: 'ESNext',
                    emptyOutDir: false,
                    rollupOptions: {
                        output: {
                            entryFileNames: '[name].js?[hash]',
                            chunkFileNames: '[name].js?[hash]',
                            assetFileNames: '[name][extname]?[hash]',
                        },
                    },
                },
            }
            switch (env.mode) {
                case 'extension-editor':
                    {
                        common.build.outDir = 'pages'
                        common.build.rollupOptions.input = {
                            index: resolve(__dirname, 'index.html'),
                        }
                    }
                    break
                case 'extension-entry':
                    {
                        common.build.outDir = 'javascript'
                        common.build.rollupOptions.input = {
                            index: resolve(
                                __dirname,
                                'src/environments/extension/entry.ts'
                            ),
                        }
                        common.publicDir = false
                        const output = common.build.rollupOptions
                            .output as OutputOptions
                        output.format = 'iife'
                    }

                    break
                default:
                    throw Error(`Unknown mode: ${env.mode}`)
            }
            return common
        },
    }
}
