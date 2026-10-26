// Real-service JavaScript SDK check, with fixed payloads supplied on stdin.
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
const [sdk, baseUrl, output] = process.argv.slice(2);
const {Valen} = await import(pathToFileURL(resolve(sdk,'src/index.js')).href);
const pkg=JSON.parse(readFileSync(resolve(sdk,'package.json'),'utf8'));
assert.equal(pkg.name,'@valen-team/sdk');
assert.equal(pkg.version,'0.2.0');
const records=[], {cases,invalid}=JSON.parse(readFileSync(0,'utf8'));
const client=Valen.remote(baseUrl,{maxRetries:0});
try {
  for (const {id,body,internal,expected} of cases) {
    const row={id,passed:false};
    try {
      const publicResult=await client.decisions(body);
      assert.deepEqual(publicResult.answers,expected.answers);
      const legacy=await client.systemOne(internal.state,internal.questions,{temperature:body.temperature});
      assert.equal(legacy.usage.output_tokens,0);
      assert.deepEqual(Object.keys(legacy.answers),body.questions.map(q=>q.name));
      for (const q of body.questions) {
        const a=publicResult.answers.find(a=>a.name===q.name), b=legacy.answers[q.name];
        const x=q.type==='predicate'?[a.probability]:a.probabilities.map(v=>v.probability);
        const y=q.type==='predicate'?[b.noul]:Object.values(b.probabilities);
        assert.equal(x.length,y.length);
        assert.ok(x.every((v,i)=>Math.abs(v-y[i])<2e-5));
      }
      Object.assign(row,{passed:true,response:publicResult});
    } catch(error) {row.error=String(error);}
    records.push(row);
  }
  for (const {id,body} of invalid) {
    const row={id,passed:false};
    try {await client.decisions(body);} catch(error) {
      row.status=error.statusCode;row.error=String(error);row.passed=error.statusCode>=400&&error.statusCode<500;
    }
    records.push(row);
  }
} finally {await client.close();}
const report={passed:records.every(r=>r.passed),validation_kind:'real_model_javascript_sdk_http',sdk_name:pkg.name,sdk_version:pkg.version,records};
writeFileSync(output,JSON.stringify(report,null,2)+'\n');
process.exitCode=report.passed?0:1;
