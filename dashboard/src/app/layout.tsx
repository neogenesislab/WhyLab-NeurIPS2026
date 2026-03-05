import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jetbrains = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "WhyLab | Causal Inference Platform",
  description: "WhyLab: Uncovering Cause and Effect in FinTech Data with Double Machine Learning.",
  icons: {
    icon: "/favicon.ico",
  },
  verification: {
    google: "ToqjqeHF5YDoqm8TPlYv7Uat8Saj_8WvIqf4Y38yR9M",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${jetbrains.variable} font-sans antialiased bg-dark-900 text-white selection:bg-brand-500 selection:text-white`}>
        <Script
          src="https://www.googletagmanager.com/gtag/js?id=G-0GVNYZLEMX"
          strategy="afterInteractive"
        />
        <Script id="ga4-init" strategy="afterInteractive">
          {`window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'G-0GVNYZLEMX');`}
        </Script>
        <div className="fixed inset-0 z-[-1] bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-brand-900/20 via-dark-900 to-dark-900 pointer-events-none" />
        {children}
      </body>
    </html>
  );
}
