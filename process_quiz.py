"""
临时脚本：处理UMU全部215道题，调用后端批量API
"""
import asyncio, json, time, sys
from pathlib import Path

sys.path.insert(0, 'backend')
from auto_quiz import QuestionParser

API = "http://localhost:8000"

async def main():
    # Parse questions
    qs = QuestionParser.parse('umu_questions_full.txt')
    print(f'Total: {len(qs)} questions')

    # Split into chunks of 30 for batch API
    CHUNK = 30
    CONCURRENT = 5
    chunks = [qs[i:i+CHUNK] for i in range(0, len(qs), CHUNK)]
    results = [None] * len(chunks)

    import httpx
    sem = asyncio.Semaphore(CONCURRENT)

    async def process_chunk(ci):
        async with sem:
            chunk = chunks[ci]
            body = {
                'questions': [{'question': q.text, 'question_type': 'unknown'} for q in chunk]
            }
            t0 = time.time()
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f'{API}/api/answer-batch', json=body)
                data = resp.json()
            elapsed = time.time() - t0
            answered = sum(1 for r in data.get('results', []) if r.get('success'))
            results[ci] = data.get('results', [])
            print(f'  Chunk {ci+1}/{len(chunks)}: {answered}/{len(chunk)} OK, {elapsed:.1f}s')

    # Run chunks concurrently
    start = time.time()
    tasks = [process_chunk(i) for i in range(len(chunks))]
    await asyncio.gather(*tasks)

    # Flatten results
    all_results = []
    for i, chunk_results in enumerate(results):
        if chunk_results:
            for j, r in enumerate(chunk_results):
                idx = i * CHUNK + j + 1
                answer = r.get('answer', '') if r.get('success') else '[FAIL]'
                all_results.append((idx, answer, r.get('success', False)))

    total_time = time.time() - start
    ok = sum(1 for _, _, s in all_results if s)
    fail = len(all_results) - ok

    # Save compact answers
    lines = []
    for idx, answer, success in sorted(all_results, key=lambda x: x[0]):
        if success:
            lines.append(f'#{idx:03d}  {answer}')
        else:
            lines.append(f'#{idx:03d}  [FAIL] {answer}')

    Path('umu_answers_compact.txt').write_text('\n'.join(lines), encoding='utf-8')

    # Save detailed
    detail = []
    detail.append(f'UMU刷题答案 | {len(qs)}题 | {total_time:.0f}秒 | 成功{ok} 失败{fail}')
    detail.append('='*50)
    for idx, answer, success in sorted(all_results, key=lambda x: x[0]):
        status = 'OK' if success else 'FAIL'
        q_text = qs[idx-1].text[:80] if idx <= len(qs) else '?'
        detail.append(f'[{status}] #{idx:03d}  {answer}')
        detail.append(f'       Q: {q_text}')
        detail.append('')
    Path('umu_answers_detail.txt').write_text('\n'.join(detail), encoding='utf-8')

    # Print all answers
    print(f'\n{"="*50}')
    print(f'Done! {len(qs)} questions, {total_time:.0f}s, {ok} OK, {fail} failed')
    print(f'{"="*50}')
    for idx, answer, success in sorted(all_results, key=lambda x: x[0]):
        status = '✅' if success else '❌'
        print(f'{status} #{idx:03d}  {answer}')

    print(f'\nFiles saved: umu_answers_compact.txt, umu_answers_detail.txt')

if __name__ == '__main__':
    asyncio.run(main())
