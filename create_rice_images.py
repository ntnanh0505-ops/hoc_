import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.makedirs('Ảnh', exist_ok=True)
os.makedirs('assets/rice_images', exist_ok=True)

varieties = {
    'Cammeo': {
        'title': 'GIỐNG LÚA CAMMEO',
        'color': '#D97706',
        'bg': '#FEF3C7',
        'desc': 'Hạt to, hình bầu dục, kích thước lớn',
        'grain_w': 0.44,
        'grain_h': 0.85,
        'tag': 'Hạt to - Độ nở cao',
        'details': [
            'Chiều dài (Major): ~6.5 - 7.0 mm',
            'Chiều rộng (Minor): ~3.0 - 3.4 mm',
            'Độ bầu dục: Rất rõ nét',
            'Nguồn gốc: Thổ Nhĩ Kỳ / Ý'
        ]
    },
    'Gonen': {
        'title': 'GIỐNG LÚA GONEN',
        'color': '#2563EB',
        'bg': '#DBEAFE',
        'desc': 'Hạt tròn đều, kích thước ngắn/vừa',
        'grain_w': 0.46,
        'grain_h': 0.70,
        'tag': 'Hạt cân đối - Độ dẻo vừa',
        'details': [
            'Chiều dài (Major): ~5.8 - 6.2 mm',
            'Chiều rộng (Minor): ~2.8 - 3.1 mm',
            'Độ tròn: Cân đối, đều',
            'Nguồn gốc: Thổ Nhĩ Kỳ'
        ]
    },
    'Jasmine': {
        'title': 'GIỐNG LÚA JASMINE',
        'color': '#059669',
        'bg': '#D1FAE5',
        'desc': 'Hạt thon dài mảnh, tỷ lệ dài/rộng lớn',
        'grain_w': 0.28,
        'grain_h': 0.95,
        'tag': 'Hạt thon dài - Gạo thơm',
        'details': [
            'Chiều dài (Major): ~7.2 - 7.8 mm',
            'Chiều rộng (Minor): ~2.0 - 2.3 mm',
            'Độ mảnh: Cao nhất (thon dài)',
            'Nguồn gốc: Đông Nam Á / Thơm'
        ]
    },
    'Osmancik': {
        'title': 'GIỐNG LÚA OSMANCIK',
        'color': '#7C3AED',
        'bg': '#EDE9FE',
        'desc': 'Hạt kích thước trung bình, hơi tròn, độ bóng cao',
        'grain_w': 0.48,
        'grain_h': 0.72,
        'tag': 'Hạt hơi tròn - Độ bóng cao',
        'details': [
            'Chiều dài (Major): ~5.9 - 6.3 mm',
            'Chiều rộng (Minor): ~3.1 - 3.3 mm',
            'Độ lệch tâm: Thấp nhất (rất tròn)',
            'Nguồn gốc: Phổ biến tại Thổ Nhĩ Kỳ'
        ]
    }
}

for name, info in varieties.items():
    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=200)
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FFFFFF')
    
    card = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle='round,pad=0.03',
                                  facecolor=info['bg'], edgecolor=info['color'], linewidth=2.5, alpha=0.35)
    ax.add_patch(card)
    
    ax.text(0.5, 0.90, info['title'], fontsize=13, fontweight='bold', color=info['color'],
            ha='center', va='center')
    ax.text(0.5, 0.83, info['desc'], fontsize=9, fontstyle='italic', color='#334155',
            ha='center', va='center')
            
    center_x, center_y = 0.28, 0.45
    gw = info['grain_w'] * 0.32
    gh = info['grain_h'] * 0.38
    
    shadow = patches.Ellipse((center_x + 0.012, center_y - 0.012), gw, gh, angle=15,
                             facecolor='#000000', alpha=0.12)
    ax.add_patch(shadow)
    
    grain = patches.Ellipse((center_x, center_y), gw, gh, angle=15,
                            facecolor='#FFFDF0', edgecolor=info['color'], linewidth=2.5)
    ax.add_patch(grain)
    
    shine = patches.Ellipse((center_x - 0.012, center_y + 0.02), gw*0.5, gh*0.7, angle=15,
                            facecolor='#FFFFFF', alpha=0.6)
    ax.add_patch(shine)
    
    tag_str = info['tag']
    ax.text(center_x, 0.16, f"Hình thái hạt gạo\n({tag_str})", fontsize=8.5, fontweight='bold',
            color='#475569', ha='center', va='center')
            
    tx = 0.54
    ty = 0.62
    ax.text(tx, 0.68, 'ĐẶC TRƯNG HÌNH THÁI:', fontsize=9.5, fontweight='bold', color='#1E293B')
    for idx, d in enumerate(info['details']):
        ax.text(tx, ty - idx * 0.09, f'• {d}', fontsize=8.2, color='#334155')
        
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    p1 = os.path.join('Ảnh', f'{name}.png')
    p2 = os.path.join('assets', 'rice_images', f'{name}.png')
    plt.savefig(p1, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(p2, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

print('Done creating images.')
