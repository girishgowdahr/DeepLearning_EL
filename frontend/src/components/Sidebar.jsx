import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Flame,
  LayoutDashboard,
  Video,
  Image as ImageIcon,
  Film,
  Cpu,
  Scan,
  History,
  BarChart3,
  Info,
  Settings,
  Smartphone
} from 'lucide-react';

const navItems = [
  { path: '/', label: 'Overview', icon: LayoutDashboard },
  { path: '/live', label: 'Live Camera', icon: Video },
  { path: '/image', label: 'Image Detection', icon: ImageIcon },
  { path: '/video', label: 'Video Detection', icon: Film },
  { path: '/cnn', label: 'CNN Classifier', icon: Cpu },
  { path: '/yolo', label: 'YOLO Detection', icon: Scan },
  { path: '/history', label: 'History & Logs', icon: History },
  { path: '/analytics', label: 'Analytics', icon: BarChart3 },
  { path: '/models', label: 'Model Info', icon: Info },
  { path: '/settings', label: 'Settings', icon: Settings },
  { path: '/camera', label: 'Phone Stream', icon: Smartphone }
];

export function Sidebar() {
  return (
    <aside className="app-sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">
          <Flame size={20} />
        </div>
        <div>
          <div className="sidebar-brand-title">ForestGuard AI</div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Wildfire Early Warning</div>
        </div>
      </div>
      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              end={item.path === '/'}
            >
              <Icon size={18} />
              <span className="nav-label">{item.label}</span>
            </NavLink>
          );
        })}
      </nav>
      <div style={{ padding: '16px', borderTop: '1px solid var(--border-color)', fontSize: '0.7rem', color: 'var(--text-dim)', textAlign: 'center' }}>
        Academic Prototype v1.0<br/>Not certified for safety
      </div>
    </aside>
  );
}
