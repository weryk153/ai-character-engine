import {build} from 'esbuild';
import {resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
const test=process.argv.includes('--test');
await build({entryPoints:[test?'adapter-regression.ts':'main.ts'],bundle:true,format:test?'cjs':'esm',platform:test?'node':'browser',outfile:test?'adapter-regression.cjs':'app.js',nodePaths:[resolve('node_modules')]});
if(test)process.exit(spawnSync(process.execPath,['adapter-regression.cjs'],{stdio:'inherit'}).status??1);
