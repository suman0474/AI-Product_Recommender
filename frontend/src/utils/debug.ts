/**
 * Debug utility for environment-aware logging.
 * Only logs in development mode to prevent console spam in production.
 *
 * Usage:
 *   import { debug } from '@/utils/debug';
 *   debug.log('Message', data);  // Only logs in dev
 *   debug.warn('Warning');       // Only warns in dev
 *   debug.error('Error');        // Always logs errors
 */

const isDev = import.meta.env.DEV;

export const debug = {
  /**
   * Log a message (development only)
   */
  log: (...args: any[]) => {
    if (isDev) {
      console.log(...args);
    }
  },

  /**
   * Log a warning (development only)
   */
  warn: (...args: any[]) => {
    if (isDev) {
      console.warn(...args);
    }
  },

  /**
   * Log an error (always - errors should be visible in production too)
   */
  error: (...args: any[]) => {
    console.error(...args);
  },

  /**
   * Log with a specific tag prefix (development only)
   */
  tagged: (tag: string) => ({
    log: (...args: any[]) => {
      if (isDev) {
        console.log(`[${tag}]`, ...args);
      }
    },
    warn: (...args: any[]) => {
      if (isDev) {
        console.warn(`[${tag}]`, ...args);
      }
    },
    error: (...args: any[]) => {
      console.error(`[${tag}]`, ...args);
    },
  }),

  /**
   * Log timing information (development only)
   */
  time: (label: string) => {
    if (isDev) {
      console.time(label);
    }
  },

  timeEnd: (label: string) => {
    if (isDev) {
      console.timeEnd(label);
    }
  },
};

export default debug;
