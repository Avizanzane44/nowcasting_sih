import codecs

with open('dashboard/dashboard.html', 'rb') as f:
    content = f.read()

try:
    text = content.decode('utf-8')
    if text.startswith('\ufeff'):
        text = text[1:]
    fixed_text = text.encode('latin-1').decode('utf-8')
    
    if '<meta charset="UTF-8" />' in fixed_text:
        fixed_text = fixed_text.replace('<meta charset="UTF-8" />', '<meta charset="utf-8">')
    else:
        # Just in case it's not found, maybe replace the `<head>` tag
        fixed_text = fixed_text.replace('<head>', '<head>\n  <meta charset="utf-8">')
    
    with open('dashboard/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(fixed_text)
    print('Fixed double-encoding using latin-1 successfully.')
except Exception as e:
    print('Error:', e)
