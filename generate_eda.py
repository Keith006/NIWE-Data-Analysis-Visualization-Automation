import os
import re
import json
import zipfile
import smtplib
import argparse
import time
import warnings
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from fpdf import FPDF
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, 'Main IP')
SPEED_COLORS = {'150m_A': '#e74c3c', '150m_B': '#c0392b', '120m': '#2980b9', '100m': '#27ae60', '80m': '#8e44ad', '50m': '#f39c12', '30m': '#1abc9c', '10m': '#e67e22'}
DIR_COLORS = {'146m': '#e74c3c', '118m': '#2980b9', '98m': '#27ae60', '48m': '#f39c12'}
WIND_ROSE_COLORS = {'146m': '#e74c3c', '118m': '#2980b9', '98m': '#27ae60', '48m': '#f39c12'}
CORR_PAIR_COLORS = ['#e74c3c', '#2980b9', '#27ae60', '#8e44ad', '#f39c12', '#1abc9c', '#e67e22', '#3498db', '#c0392b', '#2ecc71', '#9b59b6', '#d35400', '#16a085', '#c0392b', '#2c3e50']
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11, 'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight', 'figure.facecolor': 'white', 'axes.facecolor': '#fafafa', 'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--'})
def parse_speed_columns(columns):
    speed_cols = {}
    for col in columns:
        if 'wind_speed;Avg' in col:
            m = re.search('(\\d+)m', col)
            if m:
                height = m.group(1) + 'm'
                if '150' in col:
                    if 'C1' in col or '135 Deg' in col.split(';')[0].split('-')[0]:
                        key = height + '_A'
                    elif 'C2' in col or '315 Deg' in col.split(';')[0].split('-')[0]:
                        key = height + '_B'
                    else:
                        key = height + '_A'
                else:
                    key = height
                speed_cols[key] = col
    return speed_cols
def parse_speed_sd_columns(columns):
    sd_cols = {}
    for col in columns:
        if 'wind_speed;StdDev' in col:
            m = re.search('(\\d+)m', col)
            if m:
                height = m.group(1) + 'm'
                if '150' in col:
                    if 'C1' in col or ('135' in col.split(';')[0] and 'C2' not in col):
                        key = height + '_A'
                    else:
                        key = height + '_B'
                else:
                    key = height
                sd_cols[key] = col
    return sd_cols
def parse_speed_max_columns(columns):
    max_cols = {}
    for col in columns:
        if 'wind_speed;Max' in col:
            m = re.search('(\\d+)m', col)
            if m:
                height = m.group(1) + 'm'
                if '150' in col:
                    if 'C1' in col or ('135' in col.split(';')[0] and 'C2' not in col):
                        key = height + '_A'
                    else:
                        key = height + '_B'
                else:
                    key = height
                max_cols[key] = col
    return max_cols
def parse_direction_columns(columns):
    dir_cols = {}
    for col in columns:
        if 'wind_direction;Avg' in col:
            m = re.search('(\\d+)m', col)
            if m:
                height = m.group(1) + 'm'
                dir_cols[height] = col
    return dir_cols
def parse_direction_sd_columns(columns):
    sd_cols = {}
    for col in columns:
        if 'wind_direction;StdDev' in col:
            m = re.search('(\\d+)m', col)
            if m:
                height = m.group(1) + 'm'
                sd_cols[height] = col
    return sd_cols
def parse_temp_columns(columns):
    temp_cols = {}
    for col in columns:
        if 'temperature;Avg' in col:
            m = re.search('(\\d+)m', col)
            if m:
                height = m.group(1) + 'm'
                temp_cols[height] = col
    return temp_cols
def parse_humidity_columns(columns):
    hum_cols = {}
    for col in columns:
        if 'humidity;Avg' in col:
            m = re.search('(\\d+)m', col)
            if m:
                height = m.group(1) + 'm'
                hum_cols[height] = col
    return hum_cols
def parse_pressure_columns(columns):
    pres_cols = {}
    for col in columns:
        if 'air_pressure;Avg' in col:
            m = re.search('(\\d+)m', col)
            if m:
                height = m.group(1) + 'm'
                pres_cols[height] = col
    return pres_cols
def parse_solar_column(columns):
    for col in columns:
        if 'solar_irradiance;Avg' in col:
            return col
    return None
def plot_timeseries_speed(df, speed_cols, site_label, output_dir, suffix='', is_offset=False, df_raw=None):
    fig, ax = plt.subplots(figsize=(16, 7))
    height_order = ['150m_A', '150m_B', '120m', '100m', '80m', '50m', '30m', '10m']
    for key in height_order:
        if key in speed_cols:
            col = speed_cols[key]
            label = f"Spd {key.replace('_', ' ')}"
            color = SPEED_COLORS.get(key, '#333333')
            ax.plot(df['datetime'], df[col], color=color, linewidth=0.8, label=label, alpha=0.9)
    if is_offset and df_raw is not None:
        label_added = False
        for key in height_order:
            if key in speed_cols:
                col = speed_cols[key]
                raw_mask = df_raw[col].values < 0.5
                if raw_mask.any():
                    times_zero = df['datetime'].values[raw_mask]
                    y_vals = df[col].values[raw_mask]
                    lbl = 'Erroneous (Raw Speed ~ 0)' if not label_added else None
                    ax.scatter(times_zero, y_vals, marker='o', color='yellow', edgecolor='black', s=25, zorder=10, label=lbl)
                    label_added = True
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title(f'{site_label} — Time Series: All Wind Speed Sensors{suffix}')
    ax.legend(loc='upper right', fontsize=11, ncol=2, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.xticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fname = f"timeseries_wind_speed{suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f'  ✓ {fname}')
def plot_timeseries_direction(df, dir_cols, site_label, output_dir, suffix=''):
    fig, ax = plt.subplots(figsize=(16, 7))
    for key in sorted(dir_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True):
        col = dir_cols[key]
        label = f'Dir {key}'
        color = DIR_COLORS.get(key, '#333333')
        ax.plot(df['datetime'], df[col], color=color, linewidth=0.8, label=label, alpha=0.9)
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Wind Direction (°)')
    ax.set_title(f'{site_label} — Time Series: All Wind Direction Sensors{suffix}')
    all_dir_data = []
    for key in dir_cols:
        all_dir_data.extend(df[dir_cols[key]].dropna().values)
    if all_dir_data:
        dir_min = np.nanmin(all_dir_data)
        dir_max = np.nanmax(all_dir_data)
        dir_range = dir_max - dir_min
        if dir_range < 180:
            y_lo = max(0, dir_min - dir_range * 0.15)
            y_hi = min(360, dir_max + dir_range * 0.15)
            ax.set_ylim(y_lo, y_hi)
        else:
            ax.set_ylim(0, 360)
            ax.set_yticks(range(0, 361, 45))
    else:
        ax.set_ylim(0, 360)
        ax.set_yticks(range(0, 361, 45))
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.xticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fname = f"timeseries_wind_direction{suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f'  ✓ {fname}')
def plot_timeseries_temp(df, temp_cols, site_label, output_dir, suffix=''):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#c0392b', '#e74c3c']
    for i, (key, col) in enumerate(sorted(temp_cols.items(), key=lambda x: int(re.search('\\d+', x[0]).group()), reverse=True)):
        ax.plot(df['datetime'], df[col], color=colors[i % len(colors)], linewidth=0.9, label=f'Temp {key}')
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'{site_label} — Time Series: Temperature{suffix}')
    ax.legend(loc='upper right', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.xticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fname = f"timeseries_temperature{suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f'  ✓ {fname}')
def plot_timeseries_humidity(df, hum_cols, site_label, output_dir, suffix=''):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#2980b9', '#3498db']
    for i, (key, col) in enumerate(sorted(hum_cols.items(), key=lambda x: int(re.search('\\d+', x[0]).group()), reverse=True)):
        ax.plot(df['datetime'], df[col], color=colors[i % len(colors)], linewidth=0.9, label=f'RH {key}')
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Relative Humidity (%)')
    ax.set_title(f'{site_label} — Time Series: Relative Humidity{suffix}')
    ax.legend(loc='upper right', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.xticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fname = f"timeseries_humidity{suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f'  ✓ {fname}')
def plot_timeseries_pressure(df, pres_cols, site_label, output_dir, suffix=''):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#8e44ad', '#9b59b6']
    for i, (key, col) in enumerate(sorted(pres_cols.items(), key=lambda x: int(re.search('\\d+', x[0]).group()), reverse=True)):
        ax.plot(df['datetime'], df[col], color=colors[i % len(colors)], linewidth=0.9, label=f'Pres {key}')
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title(f'{site_label} — Time Series: Barometric Pressure{suffix}')
    ax.legend(loc='upper right', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.xticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fname = f"timeseries_pressure{suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f'  ✓ {fname}')
def plot_timeseries_solar(df, solar_col, site_label, output_dir, suffix=''):
    if solar_col is None or solar_col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['datetime'], df[solar_col], color='#f39c12', linewidth=0.9, label='GHI')
    ax.fill_between(df['datetime'], 0, df[solar_col], color='#f39c12', alpha=0.15)
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Solar Irradiance (W/m²)')
    ax.set_title(f'{site_label} — Time Series: Global Horizontal Irradiance (GHI){suffix}')
    ax.legend(loc='upper right', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.xticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fname = f"timeseries_solar_irradiance{suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f'  ✓ {fname}')
def _sanitize_fname(s):
    return s.replace(' ', '_').replace('(', '').replace(')', '')
def plot_individual_speed_correlations(df, speed_cols, site_label, output_dir, suffix='', df_raw=None, offsets=None):
    height_order = ['150m_A', '150m_B', '120m', '100m', '80m', '50m', '30m', '10m']
    available = [k for k in height_order if k in speed_cols]
    if len(available) < 2:
        return
    has_offset = df_raw is not None and offsets is not None
    pair_idx = 0
    consecutive_pairs = [(available[i], available[i + 1]) for i in range(len(available) - 1)]
    for key_x, key_y in consecutive_pairs:
        col_x = speed_cols[key_x]
        col_y = speed_cols[key_y]
        label_x = f"Spd {key_x.replace('_', ' ')}"
        label_y = f"Spd {key_y.replace('_', ' ')}"
        fig, ax = plt.subplots(figsize=(8, 7))
        if has_offset:
            x_raw = df_raw[col_x].values
            y_raw = df_raw[col_y].values
            mask_raw = np.isfinite(x_raw) & np.isfinite(y_raw)
            xr, yr = (x_raw[mask_raw], y_raw[mask_raw])
            if len(xr) > 2:
                ax.scatter(xr, yr, s=14, alpha=0.35, color='#95a5a6', edgecolors='none', label='Raw Data', zorder=2)
                sl_r, ic_r, rv_r, _, _ = stats.linregress(xr, yr)
                xl_r = np.linspace(xr.min(), xr.max(), 100)
                ax.plot(xl_r, sl_r * xl_r + ic_r, color='#7f8c8d', linewidth=1.5, linestyle='--', label=f'Raw Fit (R²={rv_r ** 2:.4f})', zorder=3)
        x = df[col_x].values
        y = df[col_y].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = (x[mask], y[mask])
        if len(x) < 3:
            plt.close(fig)
            continue
        data_label = 'Corrected Data' if has_offset else 'Data'
        color = CORR_PAIR_COLORS[pair_idx % len(CORR_PAIR_COLORS)]
        ax.scatter(x, y, s=14, alpha=0.55, color=color, edgecolors='none', label=data_label, zorder=4)
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        fit_label = f'Corrected Fit (R²={r_value ** 2:.4f})' if has_offset else f'Fit (R²={r_value ** 2:.4f})'
        ax.plot(x_line, y_line, color='#e74c3c', linewidth=2, label=fit_label, zorder=5)
        all_vals = np.concatenate([x, y])
        lims = [all_vals.min() * 0.9, all_vals.max() * 1.05]
        ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.4, label='1:1 Line', zorder=1)
        ax.set_xlabel(f'{label_x} (m/s)', fontsize=11)
        ax.set_ylabel(f'{label_y} (m/s)', fontsize=11)
        ax.set_title(f'{site_label} — {label_x} vs {label_y}{suffix}', fontsize=12)
        textstr = f'y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value ** 2:.4f}\nn = {len(x)}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        if has_offset:
            offset_lines = []
            for k in [key_x, key_y]:
                if k in offsets:
                    s = offsets[k]['slope']
                    o = offsets[k]['offset']
                    offset_lines.append(f"{k.replace('_', ' ')}: slope={s:.5f}, offset={o:.5f}")
            if offset_lines:
                offset_text = 'Offset Values:\n' + '\n'.join(offset_lines)
                ax.text(0.95, 0.05, offset_text, transform=ax.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', alpha=0.9), color='#8B4513')
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        fig.tight_layout()
        fname = f'speed_corr_{key_x}_vs_{key_y}{_sanitize_fname(suffix)}.png'
        fig.savefig(os.path.join(output_dir, fname))
        plt.close(fig)
        print(f'  ✓ {fname}')
        pair_idx += 1
def plot_individual_direction_correlations(df, dir_cols, site_label, output_dir, suffix='', df_raw=None):
    keys = sorted(dir_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True)
    if len(keys) < 2:
        return
    has_raw = df_raw is not None
    pair_idx = 0
    consecutive_pairs = [(keys[i], keys[i + 1]) for i in range(len(keys) - 1)]
    for key_x, key_y in consecutive_pairs:
        col_x = dir_cols[key_x]
        col_y = dir_cols[key_y]
        fig, ax = plt.subplots(figsize=(8, 7))
        if has_raw:
            x_raw = df_raw[col_x].values
            y_raw = df_raw[col_y].values
            mask_raw = np.isfinite(x_raw) & np.isfinite(y_raw)
            xr, yr = (x_raw[mask_raw], y_raw[mask_raw])
            if len(xr) > 2:
                ax.scatter(xr, yr, s=14, alpha=0.3, color='#95a5a6', edgecolors='none', label='Raw Data', zorder=2)
                sl_r, ic_r, rv_r, _, _ = stats.linregress(xr, yr)
                xl_r = np.linspace(0, 360, 100)
                ax.plot(xl_r, sl_r * xl_r + ic_r, color='#7f8c8d', linewidth=1.5, linestyle='--', label=f'Raw Fit (R²={rv_r ** 2:.4f})', zorder=3)
        x = df[col_x].values
        y = df[col_y].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = (x[mask], y[mask])
        if len(x) < 3:
            plt.close(fig)
            continue
        data_label = 'Corrected Data' if has_raw else 'Data'
        color = DIR_COLORS.get(key_y, CORR_PAIR_COLORS[pair_idx % len(CORR_PAIR_COLORS)])
        ax.scatter(x, y, s=14, alpha=0.55, color=color, edgecolors='none', label=data_label, zorder=4)
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        x_line = np.linspace(0, 360, 100)
        y_line = slope * x_line + intercept
        fit_label = f'Corrected Fit (R²={r_value ** 2:.4f})' if has_raw else f'Fit (R²={r_value ** 2:.4f})'
        ax.plot(x_line, y_line, color='#e74c3c', linewidth=2, label=fit_label, zorder=5)
        ax.plot([0, 360], [0, 360], 'k--', linewidth=0.8, alpha=0.4, label='1:1 Line', zorder=1)
        ax.set_xlabel(f'Dir {key_x} (°)', fontsize=11)
        ax.set_ylabel(f'Dir {key_y} (°)', fontsize=11)
        ax.set_title(f'{site_label} — Dir {key_x} vs Dir {key_y}{suffix}', fontsize=12)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        textstr = f'y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value ** 2:.4f}\nn = {len(x)}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        fig.tight_layout()
        fname = f'dir_corr_{key_x}_vs_{key_y}{_sanitize_fname(suffix)}.png'
        fig.savefig(os.path.join(output_dir, fname))
        plt.close(fig)
        print(f'  ✓ {fname}')
        pair_idx += 1
def plot_wind_rose(df, dir_cols, speed_cols, site_label, output_dir, suffix=''):
    keys = sorted(dir_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True)
    n_sectors = 16
    bin_edges = np.linspace(0, 360, n_sectors + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    theta = np.deg2rad(bin_centers)
    width = 2 * np.pi / n_sectors * 0.7
    for i, key in enumerate(keys):
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        col = dir_cols[key]
        directions = df[col].dropna().values
        if len(directions) == 0:
            plt.close(fig)
            continue
        counts, _ = np.histogram(directions, bins=bin_edges)
        pct = counts / counts.sum() * 100
        color = WIND_ROSE_COLORS.get(key, CORR_PAIR_COLORS[i % len(CORR_PAIR_COLORS)])
        theta_closed = np.append(theta, theta[0])
        pct_closed = np.append(pct, pct[0])
        ax.plot(theta_closed, pct_closed, color=color, linewidth=2, label=f'Dir {key}')
        ax.set_title(f'{site_label} — Wind Direction Occurrences ({key}){suffix}', pad=20, fontsize=13)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        fname = f'wind_rose_{key}{_sanitize_fname(suffix)}.png'
        fig.savefig(os.path.join(output_dir, fname))
        plt.close(fig)
        print(f'  ✓ {fname}')
def plot_speed_diff_vs_direction(df, speed_cols, dir_cols, site_label, output_dir, suffix='', is_offset=False, df_raw=None):
    if '150m_A' in speed_cols and '150m_B' in speed_cols:
        consecutive_pairs = [('150m_A', '150m_B')]
    else:
        return
    dir_map = {'150m_A': '146m', '150m_B': '146m', '120m': '118m', '100m': '98m', '80m': '48m', '50m': '48m', '30m': '48m', '10m': '48m'}
    for upper, lower in consecutive_pairs:
        col_upper = speed_cols[upper]
        col_lower = speed_cols[lower]
        diff = df[col_upper] - df[col_lower]
        if is_offset and df_raw is not None:
            diff_raw = df_raw[col_upper] - df_raw[col_lower]
        else:
            diff_raw = diff
        dir_key = dir_map.get(upper, '146m')
        if dir_key not in dir_cols:
            if not dir_cols:
                continue
            dir_key = list(dir_cols.keys())[0]
        col_dir = dir_cols[dir_key]
        x = df[col_dir].values
        y = diff.values
        y_raw_vals = diff_raw.values
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_raw_vals)
        x_clean = x[mask]
        y_clean = y[mask]
        y_raw_clean = y_raw_vals[mask]
        if len(x_clean) < 3:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        if is_offset:
            zero_mask = np.isclose(y_raw_clean, 0, atol=1e-06)
            x_valid = x_clean[~zero_mask]
            y_valid = y_clean[~zero_mask]
            x_zero = x_clean[zero_mask]
            y_zero = y_clean[zero_mask]
            ax.scatter(x_valid, y_valid, s=12, alpha=0.6, color=DIR_COLORS.get(dir_key, '#3498db'), edgecolors='none', label='Valid Data')
            if len(x_zero) > 0:
                ax.scatter(x_zero, y_zero, s=25, alpha=0.9, color='red', marker='x', label='Raw Diff = 0 (Incorrect)')
            ax.legend(loc='upper right')
        else:
            ax.scatter(x_clean, y_clean, s=12, alpha=0.6, color=DIR_COLORS.get(dir_key, '#3498db'), edgecolors='none')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel(f'Wind Direction {dir_key} (°)', fontsize=11)
        ax.set_ylabel(f"Speed Difference ({upper.replace('_', ' ')} - {lower.replace('_', ' ')}) (m/s)", fontsize=11)
        ax.set_title(f'{site_label} — Speed Diff vs Dir {dir_key}{suffix}', fontsize=12)
        ax.set_xlim(0, 360)
        ax.grid(True, alpha=0.3, linestyle='--')
        fname = f'speed_diff_vs_dir_{upper}_vs_{lower}{_sanitize_fname(suffix)}.png'
        fig.savefig(os.path.join(output_dir, fname))
        plt.close(fig)
        print(f'  ✓ {fname}')
def plot_speed_diff_vs_speed(df, speed_cols, site_label, output_dir, suffix='', is_offset=False, df_raw=None):
    if '150m_A' in speed_cols and '150m_B' in speed_cols:
        consecutive_pairs = [('150m_A', '150m_B')]
    else:
        return
    for upper, lower in consecutive_pairs:
        col_upper = speed_cols[upper]
        col_lower = speed_cols[lower]
        diff = df[col_upper] - df[col_lower]
        x = df[col_lower].values
        y = diff.values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        if len(x_clean) < 3:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        if is_offset and df_raw is not None:
            raw_upper = df_raw[col_upper].values[mask]
            raw_lower = df_raw[col_lower].values[mask]
            raw_mask = (raw_upper < 0.5) | (raw_lower < 0.5)
            x_valid = x_clean[~raw_mask]
            y_valid = y_clean[~raw_mask]
            x_zero = x_clean[raw_mask]
            y_zero = y_clean[raw_mask]
            ax.scatter(x_valid, y_valid, s=12, alpha=0.6, color=SPEED_COLORS.get(upper, '#8e44ad'), edgecolors='none', label='Valid Data')
            if len(x_zero) > 0:
                ax.scatter(x_zero, y_zero, s=25, alpha=0.9, color='yellow', edgecolor='black', marker='o', label='Erroneous (Raw Speed ~ 0)')
            ax.legend(loc='upper right')
        else:
            ax.scatter(x_clean, y_clean, s=12, alpha=0.6, color=SPEED_COLORS.get(upper, '#8e44ad'), edgecolors='none')
        ax.set_xlabel(f"Wind Speed {lower.replace('_', ' ')} (m/s)")
        ax.set_ylabel(f"Speed Difference: {upper.replace('_', ' ')} - {lower.replace('_', ' ')} (m/s)")
        ax.set_title(f'{site_label} — Speed Diff vs Speed {lower.replace("_", " ")}{suffix}', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        fname = f'speed_diff_vs_spd_{upper}_vs_{lower}{_sanitize_fname(suffix)}.png'
        fig.savefig(os.path.join(output_dir, fname))
        plt.close(fig)
        print(f'  ✓ {fname}')
def plot_turbulence_intensity(df, speed_cols, sd_cols, site_label, output_dir, suffix=''):
    height_order = ['150m_A', '150m_B', '120m', '100m', '80m', '50m', '30m', '10m']
    available = [k for k in height_order if k in speed_cols and k in sd_cols]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(16, 6))
    for key in available:
        avg_col = speed_cols[key]
        sd_col = sd_cols[key]
        ti = df[sd_col] / df[avg_col]
        ti = ti.replace([np.inf, -np.inf], np.nan)
        color = SPEED_COLORS.get(key, '#333333')
        ax.plot(df['datetime'], ti, color=color, linewidth=0.7, label=f"TI {key.replace('_', ' ')}", alpha=0.85)
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Turbulence Intensity (TI)')
    ax.set_title(f'{site_label} — Turbulence Intensity at All Heights{suffix}')
    ax.legend(loc='upper right', fontsize=11, ncol=2, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.xticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fname = f'turbulence_intensity{_sanitize_fname(suffix)}.png'
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f'  ✓ {fname}')
def plot_wind_power_density(df, speed_cols, temp_cols, pres_cols, site_label, output_dir, suffix=''):
    height_order = ['150m_A', '150m_B', '120m', '100m', '80m', '50m', '30m', '10m']
    available = [k for k in height_order if k in speed_cols]
    if not available:
        return
    rho = 1.225
    temp_keys = sorted(temp_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True)
    pres_keys = sorted(pres_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True)
    if temp_keys and pres_keys:
        T = df[temp_cols[temp_keys[0]]].values + 273.15
        P = df[pres_cols[pres_keys[0]]].values * 100
        R = 287.05
        rho_arr = P / (R * T)
    else:
        rho_arr = np.full(len(df), rho)
    fig, ax = plt.subplots(figsize=(16, 6))
    for key in available:
        col = speed_cols[key]
        v = df[col].values
        wpd = 0.5 * rho_arr * v ** 3
        color = SPEED_COLORS.get(key, '#333333')
        ax.plot(df['datetime'], wpd, color=color, linewidth=0.7, label=f"WPD {key.replace('_', ' ')}", alpha=0.85)
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Wind Power Density (W/m²)')
    ax.set_title(f'{site_label} — Wind Power Density at All Heights{suffix}')
    ax.legend(loc='upper right', fontsize=11, ncol=2, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.xticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fname = f'wind_power_density{_sanitize_fname(suffix)}.png'
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f'  ✓ {fname}')
def generate_statistical_summary(df, speed_cols, sd_cols, max_cols, dir_cols, dir_sd_cols, temp_cols, hum_cols, pres_cols, solar_col, site_label, output_dir, suffix=''):
    rows = []
    sno = 1
    height_order = ['150m_A', '150m_B', '120m', '100m', '80m', '50m', '30m', '10m']
    for key in height_order:
        if key in speed_cols:
            col = speed_cols[key]
            data = df[col].dropna()
            height_val = re.search('(\\d+)m', key).group(0)
            rows.append({'S.No': sno, 'Label': f"Spd {key.replace('_', ' ')}", 'Units': 'm/s', 'Height': height_val, 'Valid Data Points': len(data), 'DRR (%)': round(len(data) / len(df) * 100, 1), 'Mean': round(data.mean(), 3), 'Min': round(data.min(), 3), 'Max': round(data.max(), 3), 'Std. Dev.': round(data.std(), 3)})
            sno += 1
    for key in height_order:
        if key in sd_cols:
            col = sd_cols[key]
            data = df[col].dropna()
            height_val = re.search('(\\d+)m', key).group(0)
            rows.append({'S.No': sno, 'Label': f"Spd {key.replace('_', ' ')} SD", 'Units': 'm/s', 'Height': height_val, 'Valid Data Points': len(data), 'DRR (%)': round(len(data) / len(df) * 100, 1), 'Mean': round(data.mean(), 3), 'Min': round(data.min(), 3), 'Max': round(data.max(), 3), 'Std. Dev.': round(data.std(), 3)})
            sno += 1
    for key in height_order:
        if key in max_cols:
            col = max_cols[key]
            data = df[col].dropna()
            height_val = re.search('(\\d+)m', key).group(0)
            rows.append({'S.No': sno, 'Label': f"Spd {key.replace('_', ' ')} Gust", 'Units': 'm/s', 'Height': height_val, 'Valid Data Points': len(data), 'DRR (%)': round(len(data) / len(df) * 100, 1), 'Mean': round(data.mean(), 3), 'Min': round(data.min(), 3), 'Max': round(data.max(), 3), 'Std. Dev.': round(data.std(), 3)})
            sno += 1
    dir_keys = sorted(dir_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True)
    for key in dir_keys:
        col = dir_cols[key]
        data = df[col].dropna()
        rows.append({'S.No': sno, 'Label': f'Dir {key}', 'Units': '°', 'Height': key, 'Valid Data Points': len(data), 'DRR (%)': round(len(data) / len(df) * 100, 1), 'Mean': round(data.mean(), 1), 'Min': round(data.min(), 1), 'Max': round(data.max(), 1), 'Std. Dev.': round(data.std(), 1)})
        sno += 1
    for key in dir_keys:
        if key in dir_sd_cols:
            col = dir_sd_cols[key]
            data = df[col].dropna()
            rows.append({'S.No': sno, 'Label': f'Dir {key} SD', 'Units': '°', 'Height': key, 'Valid Data Points': len(data), 'DRR (%)': round(len(data) / len(df) * 100, 1), 'Mean': round(data.mean(), 1), 'Min': round(data.min(), 1), 'Max': round(data.max(), 1), 'Std. Dev.': round(data.std(), 1)})
            sno += 1
    if solar_col and solar_col in df.columns:
        data = df[solar_col].dropna()
        rows.append({'S.No': sno, 'Label': 'GHI', 'Units': 'W/m²', 'Height': '', 'Valid Data Points': len(data), 'DRR (%)': round(len(data) / len(df) * 100, 1), 'Mean': round(data.mean(), 1), 'Min': round(data.min(), 1), 'Max': round(data.max(), 1), 'Std. Dev.': round(data.std(), 1)})
        sno += 1
    for key in sorted(temp_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True):
        col = temp_cols[key]
        data = df[col].dropna()
        rows.append({'S.No': sno, 'Label': f'Tmp {key}', 'Units': '°C', 'Height': key, 'Valid Data Points': len(data), 'DRR (%)': round(len(data) / len(df) * 100, 1), 'Mean': round(data.mean(), 2), 'Min': round(data.min(), 2), 'Max': round(data.max(), 2), 'Std. Dev.': round(data.std(), 2)})
        sno += 1
    for key in sorted(pres_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True):
        col = pres_cols[key]
        data = df[col].dropna()
        rows.append({'S.No': sno, 'Label': f'Pres {key}', 'Units': 'hPa', 'Height': key, 'Valid Data Points': len(data), 'DRR (%)': round(len(data) / len(df) * 100, 1), 'Mean': round(data.mean(), 1), 'Min': round(data.min(), 1), 'Max': round(data.max(), 1), 'Std. Dev.': round(data.std(), 1)})
        sno += 1
    for key in sorted(hum_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True):
        col = hum_cols[key]
        data = df[col].dropna()
        rows.append({'S.No': sno, 'Label': f'RH {key}', 'Units': '%', 'Height': key, 'Valid Data Points': len(data), 'DRR (%)': round(len(data) / len(df) * 100, 1), 'Mean': round(data.mean(), 1), 'Min': round(data.min(), 1), 'Max': round(data.max(), 1), 'Std. Dev.': round(data.std(), 1)})
        sno += 1
    for key in height_order:
        if key in speed_cols and key in sd_cols:
            avg_data = df[speed_cols[key]]
            sd_data = df[sd_cols[key]]
            ti = (sd_data / avg_data).replace([np.inf, -np.inf], np.nan).dropna()
            height_val = re.search('(\\d+)m', key).group(0)
            rows.append({'S.No': sno, 'Label': f"Spd {key.replace('_', ' ')} TI", 'Units': '', 'Height': height_val, 'Valid Data Points': len(ti), 'DRR (%)': round(len(ti) / len(df) * 100, 1), 'Mean': round(ti.mean(), 4), 'Min': round(ti.min(), 4), 'Max': round(ti.max(), 4), 'Std. Dev.': round(ti.std(), 4)})
            sno += 1
    temp_keys_sorted = sorted(temp_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True)
    pres_keys_sorted = sorted(pres_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True)
    if temp_keys_sorted and pres_keys_sorted:
        T = df[temp_cols[temp_keys_sorted[0]]].values + 273.15
        P = df[pres_cols[pres_keys_sorted[0]]].values * 100
        rho_arr = P / (287.05 * T)
    else:
        rho_arr = np.full(len(df), 1.225)
    for key in height_order:
        if key in speed_cols:
            v = df[speed_cols[key]].values
            wpd = pd.Series(0.5 * rho_arr * v ** 3)
            wpd = wpd.replace([np.inf, -np.inf], np.nan).dropna()
            height_val = re.search('(\\d+)m', key).group(0)
            rows.append({'S.No': sno, 'Label': f"Spd {key.replace('_', ' ')} WPD", 'Units': 'W/m²', 'Height': height_val, 'Valid Data Points': len(wpd), 'DRR (%)': round(len(wpd) / len(df) * 100, 1), 'Mean': round(wpd.mean(), 1), 'Min': round(wpd.min(), 1), 'Max': round(wpd.max(), 1), 'Std. Dev.': round(wpd.std(), 1)})
            sno += 1
    summary_df = pd.DataFrame(rows)
    fname = f"statistical_summary{suffix.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    summary_df.to_csv(os.path.join(output_dir, fname), index=False)
    print(f'  ✓ {fname}')
    return summary_df
def apply_offset_correction(df, speed_cols, sd_cols, max_cols, offsets):
    df_corrected = df.copy()
    for key, params in offsets.items():
        slope = params['slope']
        offset = params['offset']
        if key in speed_cols:
            col = speed_cols[key]
            df_corrected[col] = df[col] * slope + offset
        if key in sd_cols:
            col = sd_cols[key]
            df_corrected[col] = df[col] * slope
        if key in max_cols:
            col = max_cols[key]
            df_corrected[col] = df[col] * slope + offset
    return df_corrected
def process_site(site_dir, site_label, csv_filename, offsets=None):
    csv_path = os.path.join(site_dir, csv_filename)
    print(f"\n{'=' * 70}")
    print(f'Processing: {site_label} — {csv_filename}')
    print(f"{'=' * 70}")
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).reset_index(drop=True)
    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    speed_cols = parse_speed_columns(df.columns)
    sd_cols = parse_speed_sd_columns(df.columns)
    max_cols = parse_speed_max_columns(df.columns)
    dir_cols = parse_direction_columns(df.columns)
    dir_sd_cols = parse_direction_sd_columns(df.columns)
    temp_cols = parse_temp_columns(df.columns)
    hum_cols = parse_humidity_columns(df.columns)
    pres_cols = parse_pressure_columns(df.columns)
    solar_col = parse_solar_column(df.columns)
    if not speed_cols:
        print(f'  ⚠ Skipping {csv_filename}: no wind_speed columns found.')
        return
    print(f'  Records: {len(df)}')
    print(f"  Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f'  Speed sensors: {list(speed_cols.keys())}')
    print(f'  Direction sensors: {list(dir_cols.keys())}')
    csv_base = os.path.splitext(csv_filename)[0]
    output_dir = os.path.join(site_dir, csv_base)
    os.makedirs(output_dir, exist_ok=True)
    print(f'\n  Generating outputs → {output_dir}')
    generate_all_plots(df, speed_cols, sd_cols, max_cols, dir_cols, dir_sd_cols, temp_cols, hum_cols, pres_cols, solar_col, site_label, output_dir, suffix='')
    generate_pdf_report(site_label, output_dir, csv_base, suffix='')
    if offsets:
        output_dir_offset = os.path.join(site_dir, f'{csv_base}_with_offset')
        os.makedirs(output_dir_offset, exist_ok=True)
        print(f'\n  Applying offset corrections...')
        print(f'  Generating offset-corrected outputs → {output_dir_offset}')
        df_corrected = apply_offset_correction(df, speed_cols, sd_cols, max_cols, offsets)
        export_cols = ['datetime']
        height_order = ['150m_A', '150m_B', '120m', '100m', '80m', '50m', '30m', '10m']
        for key in height_order:
            if key in speed_cols:
                export_cols.append(speed_cols[key])
        df_export = df[export_cols].copy()
        rename_map = {speed_cols[k]: f'Spd_{k}_Raw' for k in height_order if k in speed_cols}
        df_export.rename(columns=rename_map, inplace=True)
        for k in height_order:
            if k in speed_cols:
                col = speed_cols[k]
                df_export[f'Spd_{k}_Offset'] = df_corrected[col]
        export_path = os.path.join(output_dir_offset, 'wind_speed_raw_and_offset.csv')
        df_export.to_csv(export_path, index=False)
        print(f'  ✓ wind_speed_raw_and_offset.csv')
        generate_all_plots(df_corrected, speed_cols, sd_cols, max_cols, dir_cols, dir_sd_cols, temp_cols, hum_cols, pres_cols, solar_col, site_label, output_dir_offset, suffix=' (Offset Corrected)', df_raw=df, offsets=offsets)
        generate_pdf_report(site_label, output_dir_offset, csv_base, suffix=' (Offset Corrected)')
def plot_generic_correlations(df, cols_dict, var_name, site_label, output_dir, suffix='', unit=''):
    keys = list(cols_dict.keys())
    keys = sorted(keys, key=lambda x: int(re.search('\\d+', x).group()) if re.search('\\d+', x) else 0, reverse=True)
    if len(keys) < 2:
        return
    consecutive_pairs = [(keys[i], keys[i + 1]) for i in range(len(keys) - 1)]
    pair_idx = 0
    for key_x, key_y in consecutive_pairs:
        col_x = cols_dict[key_x]
        col_y = cols_dict[key_y]
        label_x = f"{var_name} {key_x.replace('_', ' ')}"
        label_y = f"{var_name} {key_y.replace('_', ' ')}"
        x = df[col_x].values
        y = df[col_y].values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = (x[mask], y[mask])
        if len(x) < 3:
            continue
        fig, ax = plt.subplots(figsize=(8, 7))
        color = CORR_PAIR_COLORS[pair_idx % len(CORR_PAIR_COLORS)]
        ax.scatter(x, y, s=14, alpha=0.55, color=color, edgecolors='none', label='Data')
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='#e74c3c', linewidth=2, label=f'Fit (R²={r_value ** 2:.4f})')
        all_vals = np.concatenate([x, y])
        lims = [all_vals.min() * 0.9, all_vals.max() * 1.05]
        ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.4, label='1:1 Line')
        ax.set_xlabel(f'{label_x} ({unit})', fontsize=11)
        ax.set_ylabel(f'{label_y} ({unit})', fontsize=11)
        ax.set_title(f'{site_label} — {label_x} vs {label_y}{suffix}', fontsize=12)
        textstr = f'y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value ** 2:.4f}\nn = {len(x)}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        fig.tight_layout()
        fname = f"{var_name.lower().replace(' ', '_')[:4]}_corr_{key_x}_vs_{key_y}{_sanitize_fname(suffix)}.png"
        fig.savefig(os.path.join(output_dir, fname))
        plt.close(fig)
        print(f'  ✓ {fname}')
        pair_idx += 1
def plot_wind_shear_profile(df, speed_cols, site_label, output_dir, suffix=''):
    height_order = ['150m_B', '120m', '100m', '80m', '50m', '30m', '10m']
    height_map = {'150m_B': 150, '120m': 120, '100m': 100, '80m': 80, '50m': 50, '30m': 30, '10m': 10}
    heights = []
    avg_speeds = []
    for k in height_order:
        if k in speed_cols:
            col = speed_cols[k]
            mean_spd = df[col].mean()
            if pd.notna(mean_spd):
                heights.append(height_map[k])
                avg_speeds.append(mean_spd)
    if len(heights) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(avg_speeds, heights, marker='o', markersize=8, color='#2c3e50', linewidth=2)
    try:
        log_h = np.log(heights)
        log_v = np.log(avg_speeds)
        slope, intercept, r_value, _, _ = stats.linregress(log_h, log_v)
        alpha = slope
        h_fit = np.linspace(min(heights), max(heights), 100)
        v_fit = np.exp(intercept) * h_fit ** alpha
        ax.plot(v_fit, h_fit, color='#e74c3c', linestyle='--', linewidth=2, label=f'Power Law Fit (α = {alpha:.4f})')
        ax.legend(loc='lower right', fontsize=11)
    except Exception:
        pass
    ax.set_ylabel('Height (m)', fontsize=11)
    ax.set_xlabel('Average Wind Speed (m/s)', fontsize=11)
    ax.set_title(f'{site_label} — Wind Shear Profile{suffix}', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    fname = f'wind_shear_profile{_sanitize_fname(suffix)}.png'
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print(f'  ✓ {fname}')
def generate_all_plots(df, speed_cols, sd_cols, max_cols, dir_cols, dir_sd_cols, temp_cols, hum_cols, pres_cols, solar_col, site_label, output_dir, suffix='', df_raw=None, offsets=None):
    generate_statistical_summary(df, speed_cols, sd_cols, max_cols, dir_cols, dir_sd_cols, temp_cols, hum_cols, pres_cols, solar_col, site_label, output_dir, suffix)
    is_offset = offsets is not None
    plot_timeseries_speed(df, speed_cols, site_label, output_dir, suffix, is_offset=is_offset, df_raw=df_raw)
    plot_timeseries_direction(df, dir_cols, site_label, output_dir, suffix)
    plot_timeseries_temp(df, temp_cols, site_label, output_dir, suffix)
    plot_timeseries_humidity(df, hum_cols, site_label, output_dir, suffix)
    plot_timeseries_pressure(df, pres_cols, site_label, output_dir, suffix)
    plot_timeseries_solar(df, solar_col, site_label, output_dir, suffix)
    plot_wind_rose(df, dir_cols, speed_cols, site_label, output_dir, suffix)
    is_offset = offsets is not None
    plot_speed_diff_vs_direction(df, speed_cols, dir_cols, site_label, output_dir, suffix, is_offset=is_offset, df_raw=df_raw)
    plot_speed_diff_vs_speed(df, speed_cols, site_label, output_dir, suffix, is_offset=is_offset, df_raw=df_raw)
    plot_individual_speed_correlations(df, speed_cols, site_label, output_dir, suffix=suffix, df_raw=df_raw, offsets=offsets)
    plot_individual_direction_correlations(df, dir_cols, site_label, output_dir, suffix=suffix, df_raw=df_raw)
    plot_generic_correlations(df, temp_cols, 'Temp', site_label, output_dir, suffix, '°C')
    plot_generic_correlations(df, pres_cols, 'Pres', site_label, output_dir, suffix, 'hPa')
    plot_turbulence_intensity(df, speed_cols, sd_cols, site_label, output_dir, suffix)
    generate_ti_csv(df, speed_cols, sd_cols, output_dir, suffix)
    plot_wind_power_density(df, speed_cols, temp_cols, pres_cols, site_label, output_dir, suffix)
    generate_data_availability_csv(df, speed_cols, sd_cols, max_cols, dir_cols, dir_sd_cols, temp_cols, hum_cols, pres_cols, solar_col, output_dir, suffix)
    generate_wind_shear_csv(df, speed_cols, output_dir, suffix)
    plot_wind_shear_profile(df, speed_cols, site_label, output_dir, suffix)
def generate_ti_csv(df, speed_cols, sd_cols, output_dir, suffix=''):
    height_order = ['150m_A', '150m_B', '120m', '100m', '80m', '50m', '30m', '10m']
    available = [k for k in height_order if k in speed_cols and k in sd_cols]
    if not available:
        return
    ti_df = pd.DataFrame({'datetime': df['datetime']})
    for key in available:
        avg_col = speed_cols[key]
        sd_col = sd_cols[key]
        ti = (df[sd_col] / df[avg_col]).replace([np.inf, -np.inf], np.nan)
        ti_df[f'TI_{key}'] = ti.round(6)
    fname = f'turbulence_intensity{_sanitize_fname(suffix)}.csv'
    ti_df.to_csv(os.path.join(output_dir, fname), index=False)
    print(f'  ✓ {fname}')
def generate_data_availability_csv(df, speed_cols, sd_cols, max_cols, dir_cols, dir_sd_cols, temp_cols, hum_cols, pres_cols, solar_col, output_dir, suffix=''):
    rows = []
    total = len(df)
    sno = 1
    height_order = ['150m_A', '150m_B', '120m', '100m', '80m', '50m', '30m', '10m']
    for key in height_order:
        if key in speed_cols:
            valid = df[speed_cols[key]].dropna().shape[0]
            rows.append({'S.No': sno, 'Sensor': f"Spd {key.replace('_', ' ')}", 'Height': re.search('(\\d+)m', key).group(0), 'Total Records': total, 'Valid Records': valid, 'Missing Records': total - valid, 'Data Recovery Rate (%)': round(valid / total * 100, 2)})
            sno += 1
    dir_keys = sorted(dir_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True)
    for key in dir_keys:
        valid = df[dir_cols[key]].dropna().shape[0]
        rows.append({'S.No': sno, 'Sensor': f'Dir {key}', 'Height': key, 'Total Records': total, 'Valid Records': valid, 'Missing Records': total - valid, 'Data Recovery Rate (%)': round(valid / total * 100, 2)})
        sno += 1
    for key in sorted(temp_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True):
        valid = df[temp_cols[key]].dropna().shape[0]
        rows.append({'S.No': sno, 'Sensor': f'Tmp {key}', 'Height': key, 'Total Records': total, 'Valid Records': valid, 'Missing Records': total - valid, 'Data Recovery Rate (%)': round(valid / total * 100, 2)})
        sno += 1
    for key in sorted(pres_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True):
        valid = df[pres_cols[key]].dropna().shape[0]
        rows.append({'S.No': sno, 'Sensor': f'Pres {key}', 'Height': key, 'Total Records': total, 'Valid Records': valid, 'Missing Records': total - valid, 'Data Recovery Rate (%)': round(valid / total * 100, 2)})
        sno += 1
    for key in sorted(hum_cols.keys(), key=lambda x: int(re.search('\\d+', x).group()), reverse=True):
        valid = df[hum_cols[key]].dropna().shape[0]
        rows.append({'S.No': sno, 'Sensor': f'RH {key}', 'Height': key, 'Total Records': total, 'Valid Records': valid, 'Missing Records': total - valid, 'Data Recovery Rate (%)': round(valid / total * 100, 2)})
        sno += 1
    if solar_col and solar_col in df.columns:
        valid = df[solar_col].dropna().shape[0]
        rows.append({'S.No': sno, 'Sensor': 'GHI', 'Height': '', 'Total Records': total, 'Valid Records': valid, 'Missing Records': total - valid, 'Data Recovery Rate (%)': round(valid / total * 100, 2)})
        sno += 1
    avail_df = pd.DataFrame(rows)
    fname = f'data_availability{_sanitize_fname(suffix)}.csv'
    avail_df.to_csv(os.path.join(output_dir, fname), index=False)
    print(f'  ✓ {fname}')
def generate_wind_shear_csv(df, speed_cols, output_dir, suffix=''):
    height_info = {'150m': 150, '120m': 120, '100m': 100, '80m': 80, '50m': 50, '30m': 30, '10m': 10}
    height_order = ['150m', '120m', '100m', '80m', '50m', '30m', '10m']
    available = ['150m']
    for k in ['120m', '100m', '80m', '50m', '30m', '10m']:
        if k in speed_cols:
            available.append(k)
    if len(available) < 2:
        return
    pairs = [(available[i], available[i + 1]) for i in range(len(available) - 1)]
    summary_rows = []
    sno = 1
    df_shear = df.copy()
    if '150m_A' in speed_cols and '150m_B' in speed_cols:
        df_shear['150m_avg'] = (df[speed_cols['150m_A']] + df[speed_cols['150m_B']]) / 2.0
    shear_ts = pd.DataFrame()
    shear_ts['datetime'] = df['datetime']
    shear_ts['Spd_150m'] = df_shear['150m_avg']
    for h in ['120m', '100m', '80m', '50m', '30m', '10m']:
        if h in speed_cols:
            shear_ts[f'Spd_{h}'] = df[speed_cols[h]]
    for upper, lower in pairs:
        z_upper = height_info[upper]
        z_lower = height_info[lower]
        if upper == '150m':
            v_upper = df_shear['150m_avg']
        else:
            v_upper = df[speed_cols[upper]]
        v_lower = df[speed_cols[lower]]
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha_ts = np.where((v_upper > 0.1) & (v_lower > 0.1), np.log(v_upper / v_lower) / np.log(z_upper / z_lower), np.nan)
        alpha_ts = pd.Series(alpha_ts).replace([np.inf, -np.inf], np.nan)
        shear_ts[f'alpha_{upper}_vs_{lower}'] = alpha_ts.round(6)
    fname_all = f'wind_shear_all_heights_timeseries{_sanitize_fname(suffix)}.csv'
    shear_ts.to_csv(os.path.join(output_dir, fname_all), index=False)
    print(f'  ✓ {fname_all}')
    for upper, lower in pairs:
        z_upper = height_info[upper]
        z_lower = height_info[lower]
        if upper == '150m':
            v_upper = df_shear['150m_avg'].values
        else:
            v_upper = df[speed_cols[upper]].values
        v_lower = df[speed_cols[lower]].values
        mean_upper = np.nanmean(v_upper)
        mean_lower = np.nanmean(v_lower)
        alpha = np.log(mean_upper / mean_lower) / np.log(z_upper / z_lower)
        summary_rows.append({'S.No': sno, 'Upper Height': upper, 'Lower Height': lower, 'Z_upper (m)': z_upper, 'Z_lower (m)': z_lower, 'Alpha': round(alpha, 3)})
        sno += 1
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        fname_sum = f'wind_shear_summary{_sanitize_fname(suffix)}.csv'
        summary_df.to_csv(os.path.join(output_dir, fname_sum), index=False)
        print(f'  ✓ {fname_sum}')
class EDAReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Wind Mast EDA Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
def generate_pdf_report(site_label, output_dir, csv_base, suffix=''):
    pdf = EDAReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, 'Chapter 1: Main Graphs & Summary', 0, 1)
    pdf.set_font('helvetica', '', 12)
    summary_text = 'The charts below illustrate the time series data for wind speed, direction, temperature, humidity, pressure, and solar irradiance. The wind rose diagrams show the dominant wind directions and speed distributions for each sensor height over the analyzed period.'
    pdf.multi_cell(0, 8, summary_text)
    pdf.ln(5)
    main_graphs = ['timeseries_wind_speed', 'timeseries_wind_direction', 'timeseries_temperature', 'timeseries_humidity', 'timeseries_pressure', 'timeseries_solar_irradiance']
    for g in main_graphs:
        fname = f'{g}{_sanitize_fname(suffix)}.png'
        img_path = os.path.join(output_dir, fname)
        if os.path.exists(img_path):
            pdf.set_font('helvetica', 'I', 10)
            pdf.cell(0, 8, f'Source: {csv_base}.csv', 0, 1, 'C')
            pdf.image(img_path, w=pdf.epw)
            pdf.ln(5)
    wind_rose_files = [f for f in os.listdir(output_dir) if f.startswith('wind_rose_') and f.endswith('.png') and (_sanitize_fname(suffix) in f)]
    wind_rose_files.sort(key=lambda x: int(re.search('\\d+', x).group()) if re.search('\\d+', x) else 0, reverse=True)
    for file in wind_rose_files:
        img_path = os.path.join(output_dir, file)
        pdf.set_font('helvetica', 'I', 10)
        pdf.cell(0, 8, f'Source: {csv_base}.csv', 0, 1, 'C')
        pdf.image(img_path, w=pdf.epw)
        pdf.ln(5)
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, 'Chapter 2: Correlations', 0, 1)
    pdf.set_font('helvetica', '', 12)
    summary_text2 = 'This section visualizes the coherence between different sensors at consecutive heights. High R-squared values indicate strong correlation and healthy sensor performance.'
    pdf.multi_cell(0, 8, summary_text2)
    pdf.ln(5)
    corr_types = [
        ('speed_corr', 'Speed Correlation'),
        ('dir_corr', 'Direction Correlation'),
        ('temp_corr', 'Temperature Correlation'),
        ('pres_corr', 'Pressure Correlation')
    ]
    all_files = os.listdir(output_dir)
    first_corr = True
    section_index = 1
    for prefix, title_prefix in corr_types:
        type_files = [f for f in all_files if f.startswith(prefix) and f.endswith('.png') and (_sanitize_fname(suffix) in f)]
        if not type_files:
            continue
        def sort_key(x):
            m = re.search(r'(\d+)', x)
            return int(m.group(1)) if m else 0
        type_files.sort(key=sort_key, reverse=True)
        
        first_in_section = True
        for file in type_files:
            if not first_corr:
                pdf.add_page()
            first_corr = False
            
            if first_in_section:
                pdf.set_font('helvetica', 'B', 14)
                pdf.cell(0, 10, f'2.{section_index} {title_prefix}', 0, 1, 'L')
                pdf.ln(2)
                first_in_section = False

            parts = file.replace(prefix + '_', '').split('.png')[0]
            if _sanitize_fname(suffix):
                parts = parts.replace(_sanitize_fname(suffix), '')
            parts = parts.replace('_vs_', ' vs ').replace('_', ' ')
            
            img_path = os.path.join(output_dir, file)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 8, f'{parts}', 0, 1, 'C')
            pdf.set_font('helvetica', 'I', 10)
            pdf.cell(0, 8, f'Source: {csv_base}.csv', 0, 1, 'C')
            pdf.image(img_path, w=pdf.epw * 0.8, x=(pdf.w - pdf.epw * 0.8) / 2)
            pdf.ln(5)
            
        section_index += 1

    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, 'Chapter 3: Speed Difference', 0, 1)
    pdf.set_font('helvetica', '', 12)
    summary_text_spd_diff = 'This section presents the speed difference between main anemometers against wind speed and wind direction.'
    pdf.multi_cell(0, 8, summary_text_spd_diff)
    pdf.ln(5)
    speed_diff_files = [f for f in all_files if f.startswith('speed_diff_vs') and f.endswith('.png') and _sanitize_fname(suffix) in f]
    spd_vs_spd_files = [f for f in speed_diff_files if 'vs_spd' in f]
    spd_vs_dir_files = [f for f in speed_diff_files if 'vs_dir' in f]
    for file in sorted(spd_vs_spd_files) + sorted(spd_vs_dir_files):
        img_path = os.path.join(output_dir, file)
        pdf.set_font('helvetica', 'I', 10)
        pdf.cell(0, 8, f'Source: {csv_base}.csv', 0, 1, 'C')
        pdf.image(img_path, w=pdf.epw * 0.8, x=(pdf.w - pdf.epw * 0.8) / 2)
        pdf.ln(5)

    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, 'Chapter 4: Data Summary', 0, 1)
    pdf.set_font('helvetica', '', 12)
    summary_text3 = 'The tables below provide a statistical summary of the data, including availability (recovery rates) and average metrics.'
    pdf.multi_cell(0, 8, summary_text3)
    pdf.ln(5)
    csvs = ['statistical_summary', 'data_availability', 'wind_shear_summary']
    for c in csvs:
        fname = f'{c}{_sanitize_fname(suffix)}.csv'
        csv_path = os.path.join(output_dir, fname)
        if os.path.exists(csv_path):
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, fname, 0, 1)
            pdf.set_font('helvetica', '', 7)
            try:
                df_csv = pd.read_csv(csv_path)
                df_csv.columns = [str(c).replace('α', 'alpha') for c in df_csv.columns]
                text_content = df_csv.head(30).to_string(index=False)
                text_content = text_content.replace('α', 'alpha')
                pdf.set_font('courier', '', 6)
                pdf.multi_cell(0, 3, text_content)
                pdf.ln(5)
                if len(df_csv) > 30:
                    pdf.cell(0, 5, f'... and {len(df_csv) - 30} more rows.', 0, 1)
            except Exception as e:
                safe_err = str(e).encode('latin-1', 'replace').decode('latin-1')
                pdf.cell(0, 10, f'Error rendering table: {safe_err}', 0, 1)
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, 'Chapter 5: Other Analysis', 0, 1)
    pdf.set_font('helvetica', '', 12)
    summary_text4 = 'This section contains additional analysis such as turbulence intensity distributions and the wind shear profile.'
    pdf.multi_cell(0, 8, summary_text4)
    pdf.ln(5)
    other_graphs = ['turbulence_intensity', 'wind_shear_profile', 'wind_power_density']
    for g in other_graphs:
        fname = f'{g}{_sanitize_fname(suffix)}.png'
        img_path = os.path.join(output_dir, fname)
        if os.path.exists(img_path):
            pdf.set_font('helvetica', 'I', 10)
            pdf.cell(0, 8, f'Source: {csv_base}.csv', 0, 1, 'C')
            pdf.image(img_path, w=pdf.epw)
            pdf.ln(5)
    report_fname = f'{csv_base}{_sanitize_fname(suffix)}.pdf'
    pdf.output(os.path.join(output_dir, report_fname))
    print(f'  ✓ {report_fname}')
def discover_sites(data_dir):
    sites = []
    for entry in sorted(os.listdir(data_dir)):
        site_path = os.path.join(data_dir, entry)
        if not os.path.isdir(site_path) or entry.startswith('.'):
            continue
        csvs = []
        for f in sorted(os.listdir(site_path)):
            if not f.lower().endswith('.csv'):
                continue
            try:
                test_df = pd.read_csv(os.path.join(site_path, f), nrows=2, on_bad_lines='skip')
                if any(('wind_speed' in str(c) for c in test_df.columns)):
                    csvs.append(f)
            except Exception:
                continue
        if not csvs:
            continue
        label = re.sub('^\\d+\\.\\s*', '', entry)
        offsets_path = os.path.join(site_path, 'offsets.json')
        offsets = None
        if os.path.exists(offsets_path):
            with open(offsets_path) as fp:
                offsets = json.load(fp)
            print(f'  📋 Loaded offsets.json for {label} ({len(offsets)} heights)')
        sites.append({'folder': entry, 'label': label, 'csvs': csvs, 'offsets': offsets})
    return sites
def send_email_report(data_dir, site, csv_file, config_path=None):
    if config_path is None:
        config_path = os.path.join(BASE_DIR, 'email_config.json')
    if not os.path.exists(config_path):
        print(f'\n⚠ email_config.json not found at {config_path}')
        print('  Skipping email. Create email_config.json with your SMTP settings.')
        return False
    with open(config_path) as f:
        cfg = json.load(f)
    required = ['smtp_server', 'smtp_port', 'sender_email', 'sender_password', 'recipients']
    for key in required:
        if key not in cfg or (isinstance(cfg[key], str) and cfg[key].startswith('your-')):
            print(f"\n⚠ email_config.json has placeholder value for '{key}'. Update it first.")
            return False
    csv_base = os.path.splitext(csv_file)[0]
    site_path = os.path.join(data_dir, site['folder'])
    output_dir_names = [csv_base, f'{csv_base}_with_offset']
    pdfs_to_attach = []
    for out_dir_name in output_dir_names:
        out_path = os.path.join(site_path, out_dir_name)
        if os.path.isdir(out_path):
            for root, dirs, files in os.walk(out_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdfs_to_attach.append(os.path.join(root, file))
    print(f'  📄 Found {len(pdfs_to_attach)} PDF(s) to attach:')
    for p in pdfs_to_attach:
        pdf_size = os.path.getsize(p) / (1024 * 1024)
        print(f'     - {os.path.basename(p)} ({pdf_size:.1f} MB)')
    msg = MIMEMultipart('mixed')
    msg['From'] = cfg['sender_email']
    msg['To'] = ', '.join(cfg['recipients'])
    if cfg.get('cc_recipients'):
        msg['Cc'] = ', '.join(cfg['cc_recipients'])
    msg['Subject'] = f"Wind Mast EDA Report — {site['label']} — {csv_base}"
    body_text = 'Dear Sir/Madam,\n\nThis is an automated notification from the Wind Resource Assessment Division, National Institute of Wind Energy (NIWE).\n\nWe are pleased to inform you that the Exploratory Data Analysis (EDA) process has been completed successfully.\n\nThe uploaded data files have been processed, and the corresponding visualizations and analytical reports have been generated successfully. The generated reports are attached to this email for your review and reference.\n\nAny additional observations, comments, or recommendations, if applicable, will be communicated separately in a subsequent email.\n\nThis is a system-generated email. Please do not reply to this message.\n\nThank you.\n\nKind regards,\n\nWind Resource Assessment Division\nNational Institute of Wind Energy (NIWE)\n'
    msg.attach(MIMEText(body_text, 'plain'))
    total_attach_size = 0
    for p in pdfs_to_attach:
        total_attach_size += os.path.getsize(p) / (1024 * 1024)
    if total_attach_size > 25:
        print(f'  ⚠ Total attachments are {total_attach_size:.1f} MB — may exceed email limits.')
    for pdf_path in pdfs_to_attach:
        try:
            pdf_name = os.path.basename(pdf_path)
            with open(pdf_path, 'rb') as f:
                part = MIMEBase('application', 'pdf')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', 'attachment', filename=pdf_name)
                part.add_header('Content-Type', 'application/pdf', name=pdf_name)
                msg.attach(part)
                print(f'  ✅ Attached PDF: {pdf_name}')
        except Exception as e:
            print(f"  ⚠ Could not attach PDF '{os.path.basename(pdf_path)}': {e}")
    try:
        all_recipients = cfg.get('recipients', []) + cfg.get('cc_recipients', [])
        print(f"  📧 Sending email to {', '.join(all_recipients)}...")
        with smtplib.SMTP(cfg['smtp_server'], cfg['smtp_port']) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(cfg['sender_email'], cfg['sender_password'])
            server.send_message(msg)
        print('  ✅ Email sent successfully!')
        return True
    except Exception as e:
        print(f'  ❌ Email failed: {e}')
        return False
def is_file_stable(filepath, wait_time=3):
    if not os.path.exists(filepath):
        return False
    mtime1 = os.path.getmtime(filepath)
    size1 = os.path.getsize(filepath)
    time.sleep(wait_time)
    mtime2 = os.path.getmtime(filepath)
    size2 = os.path.getsize(filepath)
    return mtime1 == mtime2 and size1 == size2
def main():
    parser = argparse.ArgumentParser(description='Wind Mast Data EDA — Auto-discover sites and generate visualizations as a daemon.')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR, help=f'Path to the data directory containing site folders (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--email', action='store_true', help='Send results via email after processing (requires email_config.json)')
    parser.add_argument('--email-config', type=str, default=None, help='Path to email_config.json (default: <script_dir>/email_config.json)')
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return
    print('╔══════════════════════════════════════════════════════════════╗')
    print('║     Wind Mast Data — Exploratory Data Analysis Daemon      ║')
    print('╚══════════════════════════════════════════════════════════════╝')
    print(f'\n📂 Monitoring: {data_dir}')
    print('Press Ctrl+C to stop.')
    try:
        while True:
            sites = discover_sites(data_dir)
            for site in sites:
                site_dir = os.path.join(data_dir, site['folder'])
                csvs_to_process = []
                for csv_file in site['csvs']:
                    csv_base = os.path.splitext(csv_file)[0]
                    csv_output_dir = os.path.join(site_dir, csv_base)
                    if not os.path.exists(csv_output_dir):
                        csvs_to_process.append(csv_file)
                if not csvs_to_process:
                    continue
                print(f'\n======================================================================')
                print(f"New CSV(s) detected in: {site['label']}")
                print(f'  Files to process: {csvs_to_process}')
                print(f'======================================================================')
                print('  Checking if files are fully uploaded...')
                stable = True
                for csv in csvs_to_process:
                    csv_path = os.path.join(site_dir, csv)
                    if not is_file_stable(csv_path):
                        stable = False
                        break
                if not stable:
                    print('  Files are still being written. Will retry in the next cycle.')
                    continue
                if site['offsets']:
                    offset_path = os.path.join(site_dir, 'offsets.json')
                    if not is_file_stable(offset_path):
                        print('  offsets.json is still being written. Will retry in the next cycle.')
                        continue
                for csv_file in csvs_to_process:
                    process_site(site_dir, site['label'], csv_file, offsets=site['offsets'])
                    send_email_report(data_dir, site, csv_file, config_path=args.email_config)
            time.sleep(10)
    except KeyboardInterrupt:
        print('\n🛑 Stopped monitoring.')
if __name__ == '__main__':
    main()
