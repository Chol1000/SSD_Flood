import type { ReactNode } from "react";
import { Anchor, Card, Col, Grid, Row, Typography } from "antd";
import { PageHeader } from "../ui";

const { Text } = Typography;

export interface TocItem {
  id: string;
  label: string;
}

/** Shared shell for the document-style reference pages (About, How to Use,
 * Glossary, Contact): a bounded prose column beside a sticky table of
 * contents. AntD's Anchor supplies the scroll-spy that used to be a
 * hand-rolled IntersectionObserver here.
 *
 * Below `lg` the TOC is dropped rather than stacked above the article — on a
 * phone a ten-entry jump list pushes the actual content a full screen down,
 * and scrolling is the natural way through a document at that width anyway. */
export default function DocPage({
  eyebrow,
  title,
  toc,
  children,
}: {
  eyebrow: string;
  title: string;
  toc: TocItem[];
  children: ReactNode;
}) {
  const screens = Grid.useBreakpoint();
  const showToc = !!screens.lg;

  return (
    <Row gutter={[32, 0]} style={{ maxWidth: 1240 }}>
      <Col xs={24} lg={showToc ? 16 : 24} xl={17}>
        <PageHeader eyebrow={eyebrow} title={title} />
        <Card styles={{ body: { padding: "28px 32px 40px" } }}>{children}</Card>
      </Col>

      {showToc && (
        <Col lg={8} xl={7}>
          <div style={{ position: "sticky", top: 88 }}>
            <Text
              style={{
                fontSize: 11,
                fontWeight: 700,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                color: "var(--color-primary)",
                display: "block",
                marginBottom: 12,
              }}
            >
              On This Page
            </Text>
            <Anchor
              affix={false}
              offsetTop={80}
              items={toc.map((t) => ({ key: t.id, href: `#${t.id}`, title: t.label }))}
            />
          </div>
        </Col>
      )}
    </Row>
  );
}
